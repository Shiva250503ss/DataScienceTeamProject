# 🚀 Getting Started with RL Model Selector

## ✅ What We've Created

Your RL Model Selector is now ready! Here's what has been set up:

### 🎯 Core Application
✅ **Streamlit Web App** (`app_rl_selector.py`)
   - User-friendly interface for uploading datasets
   - Automatic model selection using RL
   - Training and evaluation with visualizations

✅ **RL Model Selector** (`rl_model_selector_classification_regression.py`)
   - Main reinforcement learning model selector
   - Supports both classification and regression
   - GPU-accelerated training
   - 32 meta-feature extraction

✅ **Sample Datasets**
   - `sample_classification.csv` - 3-class classification (1000 samples)
   - `sample_regression.csv` - Regression task (1000 samples)
   - `sample_customer_churn.csv` - Real-world example (500 samples)
   - `sample_house_prices.csv` - Real-world example (500 samples)

### 📚 Documentation
✅ **README.md** - Main project documentation
✅ **README_RL_SELECTOR.md** - Detailed user guide
✅ **CLEANUP_GUIDE.md** - How to remove old files
✅ **GETTING_STARTED.md** - This file

### 🛠️ Utilities
✅ **Quick Start Scripts**
   - `quick_start.bat` - For Windows
   - `quick_start.sh` - For Linux/Mac

✅ **Helper Scripts**
   - `create_sample_dataset.py` - Generate test datasets
   - `train_rl_model_selector.py` - Train the RL agent
   - `test_rl_selector.py` - Validate the RL model

---

## 🎯 Next Steps

### Option 1: Quick Start (Recommended for First-Time Users)

**For Windows:**
```bash
quick_start.bat
```

**For Linux/Mac:**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

This will:
1. Check Python installation
2. Create virtual environment (optional)
3. Install all dependencies
4. Check GPU availability
5. Create sample datasets
6. Optionally train RL models
7. Launch the Streamlit app

### Option 2: Manual Setup (For Experienced Users)

#### Step 1: Install Dependencies
```bash
pip install -r requirements_rl.txt
```

#### Step 2: Install PyTorch (Choose one)

**CPU Only:**
```bash
pip install torch
```

**GPU (CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Launch the App
```bash
streamlit run app_rl_selector.py
```

The app will open at: http://localhost:8501

---

## 🎬 How to Use the App

### 1. Upload Your Dataset
- Click "Browse files" in the Upload Data tab
- Select a CSV file with your data
- The file should have features and a target column

### 2. Select Target Column
- Choose which column you want to predict
- The system will auto-detect if it's classification or regression

### 3. Run Model Selection
- Go to the "Model Selection" tab
- Click "🚀 Run Model Selection"
- The RL agent will analyze your data and recommend top models

### 4. Train & Evaluate
- Go to the "Results" tab
- Select a recommended model
- Click "🎯 Train Selected Model"
- View performance metrics and visualizations

---

## 📊 Testing with Sample Datasets

We've created 4 sample datasets for you to test:

### 1. Basic Classification
**File:** `sample_classification.csv`
- **Task:** 3-class classification
- **Samples:** 1000
- **Features:** 20
- **Target:** `target`

### 2. Basic Regression
**File:** `sample_regression.csv`
- **Task:** Regression
- **Samples:** 1000
- **Features:** 15
- **Target:** `target`

### 3. Customer Churn Prediction
**File:** `sample_customer_churn.csv`
- **Task:** Binary classification (churn: 0 or 1)
- **Samples:** 500
- **Features:** age, tenure, charges, products, etc.
- **Target:** `churn`

### 4. House Price Prediction
**File:** `sample_house_prices.csv`
- **Task:** Regression (predict house prices)
- **Samples:** 500
- **Features:** square_feet, bedrooms, bathrooms, etc.
- **Target:** `price`

---

## ⚠️ Important: Training the RL Model

**Before first use, you need to train the RL model!**

The RL agent needs to learn which models work best. You have two options:

### Option 1: Train During Quick Start
When you run `quick_start.bat` or `quick_start.sh`, it will ask if you want to train the RL models.
- Answer "yes" to train (takes 10-30 minutes)
- Answer "no" to skip (you can train later)

### Option 2: Train Manually
```bash
# Train both classification and regression
python rl_model_selector_classification_regression.py --mode train --task both --n_datasets 150 --timesteps 50000

# Or train separately
python rl_model_selector_classification_regression.py --mode train --task classification
python rl_model_selector_classification_regression.py --mode train --task regression
```

**Training Time:**
- CPU: 30-60 minutes
- GPU: 10-20 minutes

**What it does:**
- Generates 150 synthetic datasets
- Trains PPO (Reinforcement Learning) agent
- Saves trained models to `rl_model_selector_gpu/models/`

---

## 🔧 Troubleshooting

### "No trained model found"
**Solution:** Train the RL models first (see above)

### "No GPU detected"
**Solution:**
1. Install NVIDIA drivers
2. Install CUDA Toolkit
3. Install GPU-enabled PyTorch
4. Or continue with CPU (slower but works)

### "Import Error: No module named 'xyz'"
**Solution:**
```bash
pip install -r requirements_rl.txt
```

### "Streamlit won't start"
**Solution:**
```bash
# Try a different port
streamlit run app_rl_selector.py --server.port 8502
```

### "Model selection is slow"
**Solution:**
- Use GPU instead of CPU
- Reduce number of cross-validation folds
- Use fewer models for testing

---

## 🎓 Understanding the Results

### Classification Metrics
- **Accuracy**: Overall correctness (0-1, higher is better)
- **Precision**: Correctness of positive predictions (0-1, higher is better)
- **Recall**: Coverage of actual positives (0-1, higher is better)
- **F1 Score**: Balance of precision and recall (0-1, higher is better)

### Regression Metrics
- **R² Score**: How well the model explains variance (-∞ to 1, higher is better)
- **MSE**: Mean Squared Error (0 to ∞, lower is better)
- **RMSE**: Root Mean Squared Error (0 to ∞, lower is better)
- **MAE**: Mean Absolute Error (0 to ∞, lower is better)

---

## 💡 Tips for Best Results

1. **Clean Your Data**
   - Remove duplicates
   - Handle missing values
   - Remove outliers if needed

2. **Feature Engineering**
   - Create meaningful features
   - Scale/normalize if needed
   - Encode categorical variables

3. **Sufficient Data**
   - Minimum 100 samples recommended
   - More data = better results
   - Balance classes for classification

4. **GPU Acceleration**
   - Use GPU for 10x faster training
   - Especially important for large datasets
   - Recommended for production use

5. **Interpret Results**
   - Don't just trust accuracy
   - Look at confusion matrix (classification)
   - Check residual plots (regression)
   - Understand what the model is learning

---

## 📁 Your CSV File Format

Your CSV should look like this:

```csv
feature1,feature2,feature3,...,target
1.2,3.4,5.6,...,0
2.3,4.5,6.7,...,1
3.4,5.6,7.8,...,0
...
```

**Requirements:**
- First row should be column names
- Last column (or any column) is your target
- Can have any number of features
- Can have categorical or numerical features
- Missing values are OK (will be handled)

---

## 🔮 Advanced Usage

### Python API
```python
from rl_model_selector_classification_regression import RLModelSelector
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Initialize
selector = RLModelSelector()
selector.load()

# Get recommendations
result = selector.select(df, target='target_column', top_k=3)

print(f"Task: {result['task_type']}")
print(f"Top 3 Models: {result['models']}")
print(f"Probabilities: {result['probs']}")
```

### Command Line
```bash
# Check GPU
python rl_model_selector_classification_regression.py --mode check-gpu

# Select model for dataset
python rl_model_selector_classification_regression.py --mode select --data your_data.csv --target target_col

# Run demo
python rl_model_selector_classification_regression.py --mode demo
```

---

## 🗑️ Cleanup Old Files

If you want to remove files not related to the RL Model Selector:

See: **[CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)**

This will help you:
- Identify files to keep vs remove
- Clean up the repository
- Organize old files into an archive

---

## 📞 Need Help?

1. **Check Documentation**
   - [README.md](README.md) - Overview
   - [README_RL_SELECTOR.md](README_RL_SELECTOR.md) - Detailed guide
   - [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md) - Cleanup instructions

2. **Test Your Setup**
   ```bash
   python test_rl_selector.py
   ```

3. **Check Logs**
   - Look in `rl_model_selector_gpu/logs/`
   - Check Streamlit console output

---

## ✨ What Makes This Special?

**Traditional Approach:**
- Try each model manually
- Takes hours or days
- Requires ML expertise
- Risk of suboptimal choice

**RL Model Selector:**
- ✅ Automatic selection in seconds
- ✅ Learns from 150+ datasets
- ✅ No ML expertise needed
- ✅ Consistently picks good models

---

## 🎉 You're All Set!

Your RL Model Selector is ready to use. Here's what to do:

1. ✅ Run `streamlit run app_rl_selector.py`
2. ✅ Upload `sample_classification.csv` to test
3. ✅ Select `target` as the target column
4. ✅ Click "Run Model Selection"
5. ✅ See the magic happen!

**Happy Model Selection! 🚀**

---

*For questions or issues, check the troubleshooting section or documentation.*
