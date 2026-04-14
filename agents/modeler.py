# agents/modeler.py

import pandas as pd
import numpy as np
from typing import Any, Dict, List
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Optional boosting libraries — not required for RL-recommended sklearn models
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CB = True
except ImportError:
    _HAS_CB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

from rl_selector.inference import RLModelSelector
from agents.base import BaseAgent


class ModelerAgent(BaseAgent):
    """
    Agent responsible for model training and selection.
    
    This is the FOURTH agent in the pipeline. It:
      1. Asks the RL Model Selector for top-3 model recommendations
      2. Trains each recommended model with 5-fold cross-validation
      3. Creates a Voting Ensemble combining all trained models
      4. Evaluates the ensemble with cross-validation
      5. Fits the final ensemble on all data
    
    The ensemble is mandatory because combining multiple models almost always
    outperforms any single model — it reduces variance and captures different
    patterns in the data.
    
    Owner: Manohar
    """
    
    def __init__(self):
        super().__init__("ModelerAgent")
        self.rl_selector = RLModelSelector()
        self.model_classes = self._get_model_classes()
    
    def _get_model_classes(self) -> Dict:
        """
        Map model name strings to their actual scikit-learn classes.
        The RL model recommends from sklearn models (8 clf, 9 reg).
        XGBoost/LightGBM/CatBoost are added when available as extras.
        """
        classes = {
            # === Classification Models (all 8 that RL knows about) ===
            'LogisticRegression': LogisticRegression,
            'GaussianNB': GaussianNB,
            'KNeighborsClassifier': KNeighborsClassifier,
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            # === Regression Models (all 9 that RL knows about) ===
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'SVR': SVR,
            'KNeighborsRegressor': KNeighborsRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
        }
        # Optional boosting libraries
        if _HAS_XGB:
            classes['XGBClassifier'] = XGBClassifier
            classes['XGBRegressor'] = XGBRegressor
        if _HAS_LGBM:
            classes['LGBMClassifier'] = LGBMClassifier
            classes['LGBMRegressor'] = LGBMRegressor
        if _HAS_CB:
            classes['CatBoostClassifier'] = CatBoostClassifier
            classes['CatBoostRegressor'] = CatBoostRegressor
        return classes
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Train models and create ensemble"""
        self.log("Starting model training...")

        X = state['X']
        y = state['y']
        task_type = state['task_type']
        meta_features = state.get('meta_features')

        # Ensure X is fully numeric float64 — last-resort guard.
        # Use explicit dtype checks (more reliable than try/except astype across
        # pandas/numpy versions where DTypePromotionError may not be TypeError).
        X = X.copy()
        cols_to_drop = []
        for col in list(X.columns):
            if (pd.api.types.is_datetime64_any_dtype(X[col]) or
                    pd.api.types.is_timedelta64_dtype(X[col])):
                cols_to_drop.append(col)
            elif pd.api.types.is_bool_dtype(X[col]):
                X[col] = X[col].astype(np.float64)
            elif pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(np.float64)
            else:
                cols_to_drop.append(col)
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            self.log(f"Dropped {len(cols_to_drop)} non-numeric columns before training: {cols_to_drop}")
        X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

        if X.shape[1] == 0:
            raise ValueError("No numeric features left after cleaning — cannot train.")

        self.log(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} cols, all float64")

        # =====================================================================
        # Step 1: Get model recommendations
        #   • If the user chose a specific model → use only that model
        #   • Otherwise → ask the PPO RL agent for top-3 recommendations
        # =====================================================================
        user_selected_model = state.get('user_selected_model')

        if user_selected_model and user_selected_model in self.model_classes:
            # User explicitly chose a model from the UI
            recommendations = [(user_selected_model, 1.0)]
            self.log(f"User selected model: {user_selected_model} — skipping RL selector")
        elif meta_features is not None:
            try:
                recommendations = self.rl_selector.recommend(meta_features, task_type, top_k=3)
            except Exception as e:
                self.log(f"RL selector failed ({e}), using defaults")
                recommendations = self.rl_selector._default_recommendations(task_type)
        else:
            recommendations = self.rl_selector._default_recommendations(task_type)

        self.log("Model recommendations:")
        for model_name, confidence in recommendations:
            self.log(f"  - {model_name} (confidence: {confidence:.1%})")

        # =====================================================================
        # Step 2: Train each recommended model with 5-fold CV
        # =====================================================================
        trained_models = {}
        cv_scores = {}

        for model_name, confidence in recommendations:
            self.log(f"Training {model_name}...")
            try:
                model, score = self._train_model(model_name, X, y, task_type)
                trained_models[model_name] = model
                cv_scores[model_name] = score
                self.log(f"  [OK] {model_name} CV score: {score:.4f}")
            except Exception as e:
                self.log(f"  [FAIL] {model_name} failed: {e}")

        # =====================================================================
        # Fallback cascade: try progressively simpler models until one works
        # =====================================================================
        if len(trained_models) == 0:
            fallbacks_clf = ['RandomForestClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'LogisticRegression']
            fallbacks_reg = ['RandomForestRegressor', 'DecisionTreeRegressor', 'Ridge', 'Lasso']
            fallbacks = fallbacks_clf if task_type == 'classification' else fallbacks_reg

            for fb_name in fallbacks:
                try:
                    self.log(f"Trying fallback: {fb_name}...")
                    model, score = self._train_model(fb_name, X, y, task_type)
                    trained_models[fb_name] = model
                    cv_scores[fb_name] = score
                    self.log(f"  [OK] fallback {fb_name} CV score: {score:.4f}")
                    break
                except Exception as e:
                    self.log(f"  [FAIL] fallback {fb_name}: {e}")

        if len(trained_models) == 0:
            raise RuntimeError(
                "All models failed to train. Check that your feature columns are numeric "
                "and the target column has valid values."
            )

        # =====================================================================
        # Step 3: Build ensemble from trained models
        # VotingClassifier soft voting requires predict_proba.
        # Filter to only models that support it, fall back to hard/mean voting.
        # =====================================================================
        self.log("Creating ensemble...")

        if task_type == 'classification':
            # Only include models with predict_proba for soft voting
            soft_estimators = [(n, m) for n, m in trained_models.items()
                               if hasattr(m, 'predict_proba')]
            if len(soft_estimators) >= 2:
                ensemble_model = VotingClassifier(estimators=soft_estimators, voting='soft')
            elif len(soft_estimators) == 1:
                # Only one model with predict_proba — use hard voting with all
                all_estimators = list(trained_models.items())
                ensemble_model = VotingClassifier(estimators=all_estimators, voting='hard')
            else:
                all_estimators = list(trained_models.items())
                ensemble_model = VotingClassifier(estimators=all_estimators, voting='hard')
        else:
            ensemble_model = VotingRegressor(estimators=list(trained_models.items()))

        self.log(f"Created {'VotingClassifier' if task_type == 'classification' else 'VotingRegressor'} "
                 f"with {len(trained_models)} models: {list(trained_models.keys())}")

        # =====================================================================
        # Step 4: Evaluate ensemble with cross-validation
        # =====================================================================
        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        try:
            ensemble_cv = cross_val_score(ensemble_model, X, y, cv=5, scoring=scoring,
                                          error_score=0.0)
            ensemble_score = float(ensemble_cv.mean())
            self.log(f"Ensemble CV score: {ensemble_score:.4f} (+/-{ensemble_cv.std():.4f})")
        except Exception as e:
            self.log(f"Ensemble CV failed ({e}), using mean of individual CV scores")
            ensemble_score = float(np.mean(list(cv_scores.values())))

        # =====================================================================
        # Step 5: Fit ensemble on ALL data
        # =====================================================================
        self.log("Fitting ensemble on full dataset...")
        try:
            ensemble_model.fit(X, y)
        except Exception as e:
            # Ensemble fit failed — fall back to best single model score for comparison
            self.log(f"Ensemble fit failed ({e}), will compare single models only")
            best_fb = max(cv_scores, key=cv_scores.get)
            ensemble_model = trained_models[best_fb]
            ensemble_score = cv_scores[best_fb]

        # =====================================================================
        # Step 5b: Select the best overall model (compare all 4: 3 single + ensemble)
        # =====================================================================
        all_scores = {**cv_scores, 'Ensemble': ensemble_score}
        best_model_name = max(all_scores, key=all_scores.get)
        best_model_score = all_scores[best_model_name]

        if best_model_name == 'Ensemble':
            final_model = ensemble_model
            self.log(f"Ensemble wins ({ensemble_score:.4f}) — using as final model.")
        else:
            # A single model beat the ensemble; individual models are already
            # fitted on the full dataset inside _train_model(), so use directly.
            final_model = trained_models[best_model_name]
            self.log(
                f"{best_model_name} wins "
                f"({best_model_score:.4f} > Ensemble {ensemble_score:.4f}) "
                f"— using as final prediction model."
            )

        self.log(f"\n{'='*50}")
        self.log(f"RESULTS:")
        for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            marker = " ← BEST (final model)" if name == best_model_name else ""
            self.log(f"  {name}: {score:.4f}{marker}")
        self.log(f"{'='*50}")
        
        # =====================================================================
        # Step 6: Overfitting Detection
        # =====================================================================
        overfitting_analysis = self._detect_overfitting(
            X, y, final_model, task_type, best_model_score, cv_scores
        )
        if overfitting_analysis.get('is_suspicious'):
            self.log(f"⚠️  OVERFITTING WARNING: {overfitting_analysis['reason']}")

        # =====================================================================
        # Step 7: Error Analysis (which samples does the model get wrong?)
        # =====================================================================
        error_analysis = self._perform_error_analysis(
            X, y, final_model, task_type, state.get('raw_data'), state.get('target_column')
        )
        self.log(f"Error analysis: {len(error_analysis.get('worst_samples', []))} worst predictions analyzed")

        # =====================================================================
        # Step 8: Comprehensive Metrics
        #   Classification → Accuracy, Precision, Recall, F1, ROC-AUC
        #   Regression     → R², MAE, MSE, RMSE
        # =====================================================================
        comprehensive_metrics = self._compute_comprehensive_metrics(
            X, y, final_model, task_type
        )
        self.log("Comprehensive metrics computed:")
        for metric_name_log, metric_val in comprehensive_metrics.items():
            if isinstance(metric_val, float):
                self.log(f"  {metric_name_log}: {metric_val:.4f}")

        # Update pipeline state
        state['trained_models'] = trained_models
        state['cv_scores'] = cv_scores
        state['ensemble_model'] = final_model      # best overall model (used for predictions)
        state['ensemble_score'] = ensemble_score   # voting ensemble's CV score (for display)
        state['best_model_name'] = best_model_name
        state['model_recommendations'] = recommendations
        state['overfitting_analysis'] = overfitting_analysis
        state['error_analysis'] = error_analysis
        state['comprehensive_metrics'] = comprehensive_metrics
        state['stage'] = 'modeled'
        
        return state
    
    # =========================================================================
    # STEP 2: Train a Single Model
    # =========================================================================
    
    def _train_model(self, model_name: str, X: pd.DataFrame,
                     y: pd.Series, task_type: str) -> tuple:
        """
        Train a single model with Optuna hyperparameter tuning.

        Process:
          1. Try Optuna tuning (25 trials, 90s timeout per model)
          2. If Optuna unavailable or fails, use improved defaults
          3. Run 5-fold cross-validation to get the score
          4. Fit the model on full data
          5. Return (fitted_model, cv_score)
        """
        # Try Optuna hyperparameter tuning first
        if _HAS_OPTUNA:
            try:
                model, score = self._optuna_tune(model_name, X, y, task_type)
                return model, score
            except Exception as e:
                self.log(f"  Optuna tuning failed ({e}), using defaults")

        # Fallback: improved default hyperparameters
        model_class = self.model_classes[model_name]
        params = self._get_default_params(model_name)
        model = model_class(**params)

        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)

        model.fit(X, y)
        return model, scores.mean()
    
    def _get_default_params(self, model_name: str) -> Dict:
        """
        Improved default hyperparameters for each model.

        Key improvements over naive defaults:
          - class_weight='balanced' for all classifiers (handles imbalanced data)
          - 300 trees instead of 100 (more = better for ensembles)
          - max_depth=None for RF/ExtraTrees (let trees grow fully)
          - learning_rate + subsample for boosting (reduces overfitting)
          - weights='distance' for KNN (closer neighbors matter more)
        """
        params = {
            # --- Boosting models ---
            'XGBClassifier': {
                'n_estimators': 300, 'max_depth': 6,
                'learning_rate': 0.1, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'min_child_weight': 3,
                'tree_method': 'auto', 'random_state': 42,
                'eval_metric': 'logloss', 'verbosity': 0,
            },
            'XGBRegressor': {
                'n_estimators': 300, 'max_depth': 6,
                'learning_rate': 0.1, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'min_child_weight': 3,
                'tree_method': 'auto', 'random_state': 42,
                'verbosity': 0,
            },
            'LGBMClassifier': {
                'n_estimators': 300, 'max_depth': -1,
                'learning_rate': 0.1, 'num_leaves': 31,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'class_weight': 'balanced',
                'random_state': 42, 'verbose': -1,
            },
            'LGBMRegressor': {
                'n_estimators': 300, 'max_depth': -1,
                'learning_rate': 0.1, 'num_leaves': 31,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'random_state': 42, 'verbose': -1,
            },
            'CatBoostClassifier': {
                'iterations': 300, 'depth': 6,
                'learning_rate': 0.1,
                'auto_class_weights': 'Balanced',
                'random_state': 42, 'verbose': 0,
            },
            'CatBoostRegressor': {
                'iterations': 300, 'depth': 6,
                'learning_rate': 0.1,
                'random_state': 42, 'verbose': 0,
            },
            # --- Ensemble tree models ---
            'RandomForestClassifier': {
                'n_estimators': 300, 'max_depth': None,
                'min_samples_split': 5, 'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42, 'n_jobs': -1,
            },
            'RandomForestRegressor': {
                'n_estimators': 300, 'max_depth': None,
                'min_samples_split': 5, 'min_samples_leaf': 2,
                'random_state': 42, 'n_jobs': -1,
            },
            'ExtraTreesClassifier': {
                'n_estimators': 300, 'max_depth': None,
                'min_samples_split': 5, 'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42, 'n_jobs': -1,
            },
            'ExtraTreesRegressor': {
                'n_estimators': 300, 'max_depth': None,
                'min_samples_split': 5, 'min_samples_leaf': 2,
                'random_state': 42, 'n_jobs': -1,
            },
            'GradientBoostingClassifier': {
                'n_estimators': 200, 'max_depth': 5,
                'learning_rate': 0.1, 'subsample': 0.8,
                'min_samples_split': 5,
                'random_state': 42,
            },
            'GradientBoostingRegressor': {
                'n_estimators': 200, 'max_depth': 5,
                'learning_rate': 0.1, 'subsample': 0.8,
                'min_samples_split': 5,
                'random_state': 42,
            },
            # --- Linear models ---
            'LogisticRegression': {
                'max_iter': 2000, 'C': 1.0,
                'class_weight': 'balanced',
                'solver': 'lbfgs',
                'random_state': 42,
            },
            'DecisionTreeClassifier': {
                'max_depth': None,
                'min_samples_split': 5, 'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42,
            },
            'DecisionTreeRegressor': {
                'max_depth': None,
                'min_samples_split': 5, 'min_samples_leaf': 2,
                'random_state': 42,
            },
            'Ridge': {'alpha': 1.0, 'random_state': 42},
            'Lasso': {'alpha': 0.1, 'random_state': 42},
            'ElasticNet': {'alpha': 0.1, 'l1_ratio': 0.5, 'random_state': 42},
            # --- Distance/kernel models ---
            'SVC': {
                'probability': True, 'C': 1.0,
                'class_weight': 'balanced',
                'random_state': 42,
                'cache_size': 1000,
            },
            'SVR': {'C': 1.0, 'cache_size': 1000},
            'KNeighborsClassifier': {'n_neighbors': 5, 'weights': 'distance'},
            'KNeighborsRegressor': {'n_neighbors': 5, 'weights': 'distance'},
            # --- Probabilistic models ---
            'GaussianNB': {},
        }
        return params.get(model_name, {})

    # =========================================================================
    # Optuna Hyperparameter Tuning
    # =========================================================================

    def _optuna_tune(self, model_name: str, X: pd.DataFrame,
                     y: pd.Series, task_type: str) -> tuple:
        """
        Quick Optuna hyperparameter optimization (25 trials, 90s timeout).

        This is the biggest accuracy booster — can improve scores by 10-30%
        compared to default hyperparameters. Uses 5-fold cross-validation
        as the objective to avoid overfitting.

        SVC/SVR get special treatment: fewer trials, 3-fold CV, and a
        shorter timeout because each fit is O(n² to n³).
        """
        import optuna

        model_class = self.model_classes[model_name]
        scoring = 'accuracy' if task_type == 'classification' else 'r2'

        # SVC / SVR are O(n²-n³) — limit trials to avoid long waits
        is_svm = model_name in ('SVC', 'SVR')

        # Adapt trials to dataset size (larger = slower fits)
        n_samples = X.shape[0]
        if is_svm:
            n_trials, timeout, cv_folds = 10, 60, 3
        elif n_samples > 50000:
            n_trials, timeout, cv_folds = 10, 120, 5
        elif n_samples > 10000:
            n_trials, timeout, cv_folds = 15, 90, 5
        else:
            n_trials, timeout, cv_folds = 25, 90, 5

        # Store full param dicts keyed by trial number
        trial_params = {}

        def objective(trial):
            params = self._suggest_params(trial, model_name, task_type)
            trial_params[trial.number] = params
            try:
                model = model_class(**params)
                cv = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=scoring, error_score=0.0
                )
                return float(cv.mean())
            except Exception:
                return 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout,
            show_progress_bar=False,
        )

        # Retrieve full params for the best trial
        best_params = trial_params.get(
            study.best_trial.number, self._get_default_params(model_name)
        )
        best_score = study.best_value

        self.log(
            f"  Optuna: {len(study.trials)} trials, "
            f"best score={best_score:.4f}"
        )

        # Fit final model on full data with best params
        model = model_class(**best_params)
        model.fit(X, y)

        return model, best_score

    def _suggest_params(self, trial, model_name: str, task_type: str) -> Dict:
        """Define Optuna search space for each model type."""
        is_clf = task_type == 'classification'

        if model_name in ('RandomForestClassifier', 'RandomForestRegressor'):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42, 'n_jobs': -1,
            }
            if is_clf:
                params['class_weight'] = trial.suggest_categorical(
                    'class_weight', ['balanced', None]
                )
            return params

        if model_name in ('ExtraTreesClassifier', 'ExtraTreesRegressor'):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42, 'n_jobs': -1,
            }
            if is_clf:
                params['class_weight'] = trial.suggest_categorical(
                    'class_weight', ['balanced', None]
                )
            return params

        if model_name in ('GradientBoostingClassifier', 'GradientBoostingRegressor'):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': 42,
            }

        if model_name == 'LogisticRegression':
            return {
                'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                'solver': 'lbfgs',
                'class_weight': trial.suggest_categorical(
                    'class_weight', ['balanced', None]
                ),
                'max_iter': 2000, 'random_state': 42,
            }

        if model_name == 'SVC':
            return {
                'C': trial.suggest_float('C', 0.1, 50.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'class_weight': trial.suggest_categorical(
                    'class_weight', ['balanced', None]
                ),
                'probability': True, 'random_state': 42,
                'cache_size': 1000,   # MB — larger cache = faster kernel computations
            }

        if model_name == 'SVR':
            return {
                'C': trial.suggest_float('C', 0.1, 50.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'cache_size': 1000,
            }

        if model_name == 'SVR':
            return {
                'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
            }

        if model_name in ('KNeighborsClassifier', 'KNeighborsRegressor'):
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 25, step=2),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            }

        if model_name == 'GaussianNB':
            return {
                'var_smoothing': trial.suggest_float(
                    'var_smoothing', 1e-12, 1e-6, log=True
                ),
            }

        if model_name in ('DecisionTreeClassifier', 'DecisionTreeRegressor'):
            params = {
                'max_depth': trial.suggest_categorical(
                    'max_depth', [None, 5, 10, 15, 20]
                ),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
            }
            if is_clf:
                params['class_weight'] = trial.suggest_categorical(
                    'class_weight', ['balanced', None]
                )
            return params

        if model_name == 'Ridge':
            return {
                'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
                'random_state': 42,
            }

        if model_name == 'Lasso':
            return {
                'alpha': trial.suggest_float('alpha', 0.0001, 10.0, log=True),
                'random_state': 42,
            }

        if model_name == 'ElasticNet':
            return {
                'alpha': trial.suggest_float('alpha', 0.0001, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'random_state': 42,
            }

        if model_name in ('XGBClassifier', 'XGBRegressor'):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'tree_method': 'auto', 'random_state': 42, 'verbosity': 0,
            }
            if model_name == 'XGBClassifier':
                params['eval_metric'] = 'logloss'
            return params

        if model_name in ('LGBMClassifier', 'LGBMRegressor'):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_categorical('max_depth', [-1, 5, 10, 15]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42, 'verbose': -1,
            }
            if is_clf:
                params['class_weight'] = trial.suggest_categorical(
                    'class_weight', ['balanced', None]
                )
            return params

        if model_name in ('CatBoostClassifier', 'CatBoostRegressor'):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500, step=50),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42, 'verbose': 0,
            }
            if is_clf:
                use_balanced = trial.suggest_categorical('use_balanced', [True, False])
                if use_balanced:
                    params['auto_class_weights'] = 'Balanced'
            return params

        # Fallback to improved defaults for any unknown model
        return self._get_default_params(model_name)

    # =========================================================================
    # STEP 3: Create Ensemble
    # =========================================================================
    
    def _create_ensemble(self, trained_models: Dict, task_type: str):
        """
        Create a Voting Ensemble from all trained models.
        
        How Voting Ensemble works:
          - Classification (voting='soft'):
              Each model predicts class probabilities → average them → pick highest
              Example: Model A says [0.3, 0.7], Model B says [0.4, 0.6], Model C says [0.2, 0.8]
              Average = [0.3, 0.7] → predict class 1
          
          - Regression:
              Each model predicts a number → average them
              Example: Model A says 50, Model B says 55, Model C says 48
              Average = 51
        
        Why ensemble?
          - Reduces overfitting (different models make different errors)
          - More stable predictions
          - Almost always beats any single model
        
        Args:
            trained_models: Dict of {model_name: fitted_model}
            task_type: 'classification' or 'regression'
        
        Returns:
            VotingClassifier or VotingRegressor (unfitted — will be fitted later)
        """
        estimators = [(name, model) for name, model in trained_models.items()]
        
        if task_type == 'classification':
            # 'soft' voting = average predicted probabilities (better than hard voting)
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        else:
            ensemble = VotingRegressor(estimators=estimators)
        
        self.log(f"Created {'VotingClassifier' if task_type == 'classification' else 'VotingRegressor'} "
                f"with {len(estimators)} models: {[name for name, _ in estimators]}")
        
        return ensemble
    
    # =========================================================================
    # STEP 6: Overfitting Detection
    # =========================================================================
    
    def _detect_overfitting(self, X: pd.DataFrame, y: pd.Series,
                             model, task_type: str,
                             ensemble_score: float,
                             cv_scores: Dict) -> Dict:
        """
        Detect potential overfitting — like a real DS who questions a 98% score.
        
        Checks:
          1. Suspiciously high scores (>0.98 for classification, >0.99 for regression)
          2. Train score vs CV score gap (>0.05 means overfitting)
          3. High CV variance (models unstable across folds)
          4. Score too close to 1.0 (likely data leakage or trivial task)
        
        Returns:
            Dict with overfitting analysis results
        """
        analysis = {
            'is_suspicious': False,
            'warnings': [],
            'train_score': None,
            'cv_score': ensemble_score,
            'gap': None,
            'reason': ''
        }
        
        # Check 1: Suspiciously high CV score
        threshold = 0.98 if task_type == 'classification' else 0.99
        if ensemble_score > threshold:
            analysis['is_suspicious'] = True
            analysis['warnings'].append(
                f"Score of {ensemble_score:.4f} is suspiciously high (>{threshold}). "
                f"Check for data leakage, target column in features, or trivially easy task."
            )
        
        # Check 2: Train vs CV gap
        try:
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            if hasattr(model, 'predict'):
                from sklearn.metrics import accuracy_score, r2_score
                y_pred = model.predict(X)
                if task_type == 'classification':
                    train_score = accuracy_score(y, y_pred)
                else:
                    train_score = r2_score(y, y_pred)
                
                analysis['train_score'] = round(float(train_score), 4)
                gap = train_score - ensemble_score
                analysis['gap'] = round(float(gap), 4)
                
                if gap > 0.05:
                    analysis['is_suspicious'] = True
                    analysis['warnings'].append(
                        f"Train score ({train_score:.4f}) is {gap:.4f} higher than CV score "
                        f"({ensemble_score:.4f}) — model is overfitting."
                    )
        except Exception:
            pass
        
        # Check 3: High CV variance among individual models
        if len(cv_scores) > 1:
            score_values = list(cv_scores.values())
            score_std = np.std(score_values)
            score_range = max(score_values) - min(score_values)
            
            if score_range > 0.15:
                analysis['warnings'].append(
                    f"Models have very different scores (range: {score_range:.4f}). "
                    f"Dataset might have noise or the models are inconsistent."
                )
            
            analysis['model_score_std'] = round(float(score_std), 4)
            analysis['model_score_range'] = round(float(score_range), 4)
        
        # Build summary reason
        if analysis['warnings']:
            analysis['reason'] = ' | '.join(analysis['warnings'])
        else:
            analysis['reason'] = 'No overfitting detected — scores look healthy.'
        
        return analysis
    
    # =========================================================================
    # STEP 7: Error Analysis
    # =========================================================================
    
    def _perform_error_analysis(self, X: pd.DataFrame, y: pd.Series,
                                 model, task_type: str,
                                 raw_data: pd.DataFrame = None,
                                 target_col: str = None) -> Dict:
        """
        Analyze where the model makes mistakes — like a real DS doing error analysis.
        
        For classification:
          - Which classes get confused most?
          - Which samples are misclassified?
          - Confusion matrix analysis
        
        For regression:
          - Which samples have the highest error?
          - Is there a pattern in the errors? (e.g., high errors for low values)
          - Residual distribution analysis
        """
        error_report = {
            'worst_samples': [],
            'error_patterns': [],
            'summary': ''
        }
        
        try:
            # Use cross-validated predictions to avoid evaluating on train data
            from sklearn.model_selection import cross_val_predict
            y_pred = cross_val_predict(model, X, y, cv=5)
            
            if task_type == 'classification':
                # Find misclassified samples
                from sklearn.metrics import confusion_matrix, classification_report
                
                misclassified_mask = y != y_pred
                n_errors = misclassified_mask.sum()
                error_rate = n_errors / len(y) * 100
                
                error_report['total_errors'] = int(n_errors)
                error_report['error_rate'] = round(float(error_rate), 2)
                
                # Confusion matrix
                cm = confusion_matrix(y, y_pred)
                error_report['confusion_matrix'] = cm.tolist()
                
                # Per-class error rates
                class_labels = sorted(y.unique())
                class_errors = []
                for cls in class_labels:
                    cls_mask = y == cls
                    cls_n = cls_mask.sum()
                    cls_errors_n = (misclassified_mask & cls_mask).sum()
                    cls_error_rate = cls_errors_n / cls_n * 100 if cls_n > 0 else 0
                    class_errors.append({
                        'class': str(cls),
                        'total': int(cls_n),
                        'errors': int(cls_errors_n),
                        'error_rate': round(float(cls_error_rate), 2)
                    })
                error_report['class_errors'] = class_errors
                
                # Worst class
                worst_class = max(class_errors, key=lambda x: x['error_rate'])
                error_report['summary'] = (
                    f"{error_rate:.1f}% overall error rate. "
                    f"Worst class: '{worst_class['class']}' with {worst_class['error_rate']:.1f}% error rate."
                )
                
            else:
                # Regression: analyze high-error samples
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                errors = np.abs(y.values - y_pred)
                error_report['mae'] = round(float(np.mean(errors)), 4)
                error_report['rmse'] = round(float(np.sqrt(np.mean(errors**2))), 4)
                error_report['median_error'] = round(float(np.median(errors)), 4)
                
                # Find worst predictions
                worst_idx = np.argsort(errors)[-10:][::-1]
                worst_samples = []
                for idx in worst_idx:
                    sample = {
                        'index': int(idx),
                        'actual': round(float(y.iloc[idx]), 2),
                        'predicted': round(float(y_pred[idx]), 2),
                        'error': round(float(errors[idx]), 2)
                    }
                    # Add original feature values if available
                    if raw_data is not None and target_col:
                        for col in raw_data.columns[:5]:
                            if col != target_col:
                                val = raw_data.iloc[idx][col]
                                sample[col] = str(val)
                    worst_samples.append(sample)
                error_report['worst_samples'] = worst_samples
                
                # Check error patterns: do errors correlate with target magnitude?
                y_vals = y.values.astype(float)
                low_mask = y_vals <= np.percentile(y_vals, 25)
                high_mask = y_vals >= np.percentile(y_vals, 75)
                mid_mask = ~low_mask & ~high_mask
                
                patterns = []
                for name, mask in [('low_values (Q1)', low_mask),
                                     ('mid_values (Q2-Q3)', mid_mask),
                                     ('high_values (Q4)', high_mask)]:
                    if mask.sum() > 0:
                        group_mae = np.mean(errors[mask])
                        patterns.append({
                            'group': name,
                            'n_samples': int(mask.sum()),
                            'mae': round(float(group_mae), 4)
                        })
                error_report['error_patterns'] = patterns
                
                # Summary
                worst = max(patterns, key=lambda x: x['mae']) if patterns else None
                if worst:
                    error_report['summary'] = (
                        f"MAE = {error_report['mae']:.4f}. "
                        f"Highest error for {worst['group']} (MAE = {worst['mae']:.4f}). "
                        f"Top 10 worst predictions shown below."
                    )
        except Exception as e:
            error_report['summary'] = f"Error analysis failed: {str(e)}"
        
        return error_report

    # =========================================================================
    # STEP 8: Comprehensive Metrics
    # =========================================================================

    def _compute_comprehensive_metrics(
        self, X: pd.DataFrame, y: pd.Series,
        final_model, task_type: str
    ) -> Dict:
        """
        Compute a full set of evaluation metrics using 5-fold cross-validated
        predictions (no data leakage).

        Classification:
          • Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)
          • ROC-AUC (One-vs-Rest, macro) — only when predict_proba is available

        Regression:
          • R² (coefficient of determination)
          • MAE  (Mean Absolute Error)
          • MSE  (Mean Squared Error)
          • RMSE (Root Mean Squared Error)

        Also computes per-model metrics for every trained model so the UI
        can show a comparison table.
        """
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score,
            r2_score, mean_absolute_error, mean_squared_error
        )

        metrics: Dict[str, Any] = {'task_type': task_type}

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        try:
            y_pred = cross_val_predict(final_model, X_arr, y_arr, cv=5)

            if task_type == 'classification':
                metrics['Accuracy']  = round(float(accuracy_score(y_arr, y_pred)), 4)
                metrics['Precision'] = round(float(precision_score(
                    y_arr, y_pred, average='weighted', zero_division=0)), 4)
                metrics['Recall']    = round(float(recall_score(
                    y_arr, y_pred, average='weighted', zero_division=0)), 4)
                metrics['F1-Score']  = round(float(f1_score(
                    y_arr, y_pred, average='weighted', zero_division=0)), 4)

                # ROC-AUC (requires probability estimates)
                try:
                    y_proba = cross_val_predict(
                        final_model, X_arr, y_arr, cv=5, method='predict_proba')
                    n_classes = len(np.unique(y_arr))
                    if n_classes == 2:
                        metrics['ROC-AUC'] = round(float(
                            roc_auc_score(y_arr, y_proba[:, 1])), 4)
                    else:
                        metrics['ROC-AUC'] = round(float(
                            roc_auc_score(y_arr, y_proba, multi_class='ovr',
                                         average='weighted')), 4)
                except Exception:
                    metrics['ROC-AUC'] = 'N/A'

            else:  # regression
                metrics['R²']   = round(float(r2_score(y_arr, y_pred)), 4)
                metrics['MAE']  = round(float(mean_absolute_error(y_arr, y_pred)), 4)
                metrics['MSE']  = round(float(mean_squared_error(y_arr, y_pred)), 4)
                metrics['RMSE'] = round(float(np.sqrt(
                    mean_squared_error(y_arr, y_pred))), 4)

        except Exception as e:
            metrics['error'] = str(e)

        return metrics
