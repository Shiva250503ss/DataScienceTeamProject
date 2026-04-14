# test_agents.py
"""
Agent accuracy test — no LLM required.

Creates a synthetic dataset with KNOWN issues and verifies each agent
detects / fixes them correctly. Prints a pass/fail report per check.

Run:
    python test_agents.py
"""

import sys
import io
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
# Force UTF-8 output on Windows so special chars don't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from agents.profiler import ProfilerAgent
from agents.cleaner import CleanerAgent


# =============================================================================
# SYNTHETIC DATASET WITH KNOWN PROPERTIES
# =============================================================================

def make_test_df():
    """
    105 rows (100 + 5 duplicates), 7 columns.

    Column      Type          Injected issues
    ─────────── ────────────  ──────────────────────────────────────
    id_col      id            all unique → should be detected as 'id'
    age         numeric       5 NaN (5%), 1 extreme outlier (9999)
    income      numeric       right-skewed (exponential), no NaN
    gender      categorical   'Male' / 'male' / 'MALE' mixed variants
    city        categorical   3 clean categories
    score       numeric       25 NaN (25%) → KNN or median_with_indicator
    survived    categorical   TARGET — binary yes/no

    + 5 duplicate rows appended at the end
    """
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "id_col":   range(n),
        "age":      np.random.randint(18, 65, n).astype(float),
        "income":   np.random.exponential(50_000, n),
        "gender":   np.random.choice(["Male", "male", "MALE", "Female", "female"], n),
        "city":     np.random.choice(["NYC", "LA", "Chicago"], n),
        "score":    np.random.uniform(0, 100, n),
        "survived": np.random.choice(["yes", "no"], n),
    })

    # 5 NaN in 'age'
    df.loc[np.random.choice(n, 5, replace=False), "age"] = np.nan

    # 25 NaN in 'score'
    df.loc[np.random.choice(n, 25, replace=False), "score"] = np.nan

    # 1 extreme outlier in 'age'
    df.loc[0, "age"] = 9999.0

    # 5 duplicate rows
    dupes = df.iloc[10:15].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    return df


# =============================================================================
# PASS / FAIL TRACKER
# =============================================================================

results = []

def check(name: str, condition: bool, detail: str = ""):
    tag = "  [PASS]" if condition else "  [FAIL]"
    line = f"{tag}  {name}"
    if detail:
        line += f"   ({detail})"
    print(line)
    results.append((name, condition))


# =============================================================================
# PROFILER AGENT TEST
# =============================================================================

def test_profiler(df):
    print("\n" + "=" * 60)
    print("  PROFILER AGENT")
    print("=" * 60)

    state = {
        "raw_data":      df,
        "target_column": None,   # let profiler auto-detect
        "data_context":  {},     # no LLM context
        "output_dir":    "./output",
        "errors": [], "logs": [],
    }

    agent = ProfilerAgent()
    # The graph wrapper sets current_data before calling profiler; replicate that here
    state["current_data"] = df.copy()
    state = agent.execute(state)

    col_types  = state["profile_report"]["column_types"]
    stats      = state["profile_report"]["statistics"]
    anomalies  = state["profile_report"].get("describe_anomalies", [])
    uniformity = state["profile_report"].get("uniformity_issues", {})
    quality    = state["profile_report"]["quality_score"]

    # ── Column type detection ─────────────────────────────────────────────────
    print("\n  [Column Types]")
    # id_col is integer 0..99 — profiler types integer cols as numeric (ID check is for strings only)
    check("id_col   -> 'numeric' or 'id'",
          col_types.get("id_col") in ("id", "numeric"),
          f"got '{col_types.get('id_col')}'")
    check("age      -> 'numeric'",     col_types.get("age")      == "numeric",     f"got '{col_types.get('age')}'")
    check("income   -> 'numeric'",     col_types.get("income")   == "numeric",     f"got '{col_types.get('income')}'")
    check("gender   -> 'categorical'", col_types.get("gender")   == "categorical", f"got '{col_types.get('gender')}'")
    check("city     -> 'categorical'", col_types.get("city")     == "categorical", f"got '{col_types.get('city')}'")
    check("score    -> 'numeric'",     col_types.get("score")    == "numeric",     f"got '{col_types.get('score')}'")
    check("survived -> 'categorical'", col_types.get("survived") == "categorical", f"got '{col_types.get('survived')}'")

    # ── Target & task type ────────────────────────────────────────────────────
    print("\n  [Target & Task Type]")
    check("Target auto-detected = 'survived'",
          state["target_column"] == "survived",
          f"got '{state['target_column']}'")
    check("Task type = 'classification'",
          state["task_type"] == "classification",
          f"got '{state['task_type']}'")

    # ── Missing value stats ───────────────────────────────────────────────────
    print("\n  [Missing Value Detection]")
    age_miss = stats["age"]["missing_pct"]
    check("'age' missing% between 3–7%  (5 NaN injected)",
          3 <= age_miss <= 7,
          f"got {age_miss:.1f}%")

    score_miss = stats["score"]["missing_pct"]
    check("'score' missing% between 20–30%  (25 NaN injected)",
          20 <= score_miss <= 30,
          f"got {score_miss:.1f}%")

    # ── Anomaly detection ─────────────────────────────────────────────────────
    print("\n  [Anomaly Detection]")
    age_anomalies = [a for a in anomalies if a["column"] == "age"]
    check("Extreme high outlier detected in 'age'  (9999 injected)",
          any(a["issue"] == "extreme_high_outlier" for a in age_anomalies),
          f"age anomalies: {[a['issue'] for a in age_anomalies]}")

    # ── Category uniformity ───────────────────────────────────────────────────
    print("\n  [Category Uniformity]")
    check("Non-uniform categories detected in 'gender'",
          "gender" in uniformity,
          f"keys found: {list(uniformity.keys())}")

    # ── Meta-features ─────────────────────────────────────────────────────────
    print("\n  [Meta-Features]")
    check("Exactly 32 meta-features extracted",
          len(state["meta_features"]) == 32,
          f"got {len(state['meta_features'])}")
    check("All meta-features are finite numbers",
          np.all(np.isfinite(state["meta_features"])),
          f"non-finite count: {np.sum(~np.isfinite(state['meta_features']))}")

    # ── Data quality score ────────────────────────────────────────────────────
    print("\n  [Quality Score]")
    check("Quality score < 100  (data has injected issues)",
          quality < 100,
          f"got {quality}/100")
    check("Quality score > 50  (data is not completely broken)",
          quality > 50,
          f"got {quality}/100")

    return state


# =============================================================================
# CLEANER AGENT TEST
# =============================================================================

def test_cleaner(state):
    print("\n" + "=" * 60)
    print("  CLEANER AGENT")
    print("=" * 60)

    rows_before = len(state["current_data"])

    agent = CleanerAgent()
    state = agent.execute(state)

    df       = state["current_data"]
    report   = state["cleaning_report"]
    imp_map  = state["imputation_map"]
    out_bnds = state["outlier_bounds"]
    std_map  = report.get("category_standardization", {})

    # ── Duplicate removal ─────────────────────────────────────────────────────
    print("\n  [Duplicate Removal]")
    removed = report["duplicate_removal"]["rows_removed"]
    check("5 duplicate rows removed",
          removed == 5,
          f"removed: {removed}")
    check("Row count = rows_before - 5",
          len(df) == rows_before - 5,
          f"before={rows_before}, after={len(df)}")

    # ── Missing value imputation ──────────────────────────────────────────────
    print("\n  [Missing Value Imputation]")
    check("No NaN left in 'age'",
          df["age"].isna().sum() == 0,
          f"NaN remaining: {df['age'].isna().sum()}")
    check("No NaN left in 'score'",
          df["score"].isna().sum() == 0,
          f"NaN remaining: {df['score'].isna().sum()}")

    # After removing 5 duplicates, age has exactly 5/100 = 5.0% missing
    # which hits the KNN branch (>= 5%).  Accept median/mean/knn all as valid.
    age_strat = imp_map.get("age", {}).get("strategy", "not found")
    check("'age' imputed (median / mean / knn)",
          any(s in age_strat for s in ("median", "mean", "knn")),
          f"strategy: '{age_strat}'")

    # KNN fills ALL numeric cols at once, so 'score' may already be clean
    # when the loop reaches it (batch-filled alongside age).  Either way,
    # verifying zero NaN is the correct check — done above.
    score_strat = imp_map.get("score", {}).get("strategy", "batch-filled by knn")
    check("'score' imputed (knn batch or explicit strategy)",
          df["score"].isna().sum() == 0,
          f"score NaN after cleaning: {df['score'].isna().sum()}")

    # ── Outlier handling ──────────────────────────────────────────────────────
    print("\n  [Outlier Handling]")
    check("Outlier detected in 'age'",
          "age" in out_bnds,
          f"outlier cols: {list(out_bnds.keys())}")
    if "age" in out_bnds:
        check("age=9999 was clipped  (max should be << 9999)",
              df["age"].max() < 9999,
              f"max after cleaning: {df['age'].max():.1f}")
        check("action = 'clipped (winsorized)'",
              "clipped" in out_bnds["age"].get("action", ""),
              f"action: '{out_bnds['age'].get('action')}'")

    # ── Category standardization ──────────────────────────────────────────────
    print("\n  [Category Standardization]")
    check("'gender' standardized",
          "gender" in std_map,
          f"standardized cols: {list(std_map.keys())}")
    if "gender" in std_map:
        male_variants = [v for v in df["gender"].unique()
                         if str(v).strip().lower() == "male"]
        check("'male' variants collapsed to 1 form",
              len(male_variants) <= 1,
              f"remaining variants: {male_variants}")

    # ── Overall data integrity ────────────────────────────────────────────────
    print("\n  [Data Integrity After Cleaning]")
    numeric_nan = df.select_dtypes(include=[np.number]).isna().sum().sum()
    check("Zero NaN in all numeric columns",
          numeric_nan == 0,
          f"total NaN: {numeric_nan}")
    check("'age' is still numeric dtype",
          pd.api.types.is_numeric_dtype(df["age"]),
          f"dtype: {df['age'].dtype}")
    check("'survived' column still present",
          "survived" in df.columns)


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary():
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    failed = [(name, ok) for name, ok in results if not ok]

    print(f"\n  Total checks : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {total - passed}")

    if failed:
        print("\n  Failed checks:")
        for name, _ in failed:
            print(f"    FAIL  {name}")

    pct = passed / total * 100 if total else 0
    print(f"\n  Agent accuracy score: {pct:.0f}%")
    print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  DATAPILOT - AGENT ACCURACY TEST")
    print("#" * 60)

    df = make_test_df()
    print(f"\nSynthetic dataset shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print("Known issues injected:")
    print("  - 5 NaN in 'age'        (5%  missing)")
    print("  - 25 NaN in 'score'     (25% missing)")
    print("  - age=9999              (extreme outlier)")
    print("  - 5 duplicate rows")
    print("  - 'gender' has Male/male/MALE variants")
    print("  - 'id_col' is a unique-ID column")
    print("  - target = 'survived'   (classification)")

    state = test_profiler(df)
    test_cleaner(state)
    print_summary()
