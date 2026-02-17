# 🤖 RL Model Selector - Intelligent Model Selection using Reinforcement Learning

> **Automatically select the best machine learning model for your dataset using Reinforcement Learning (PPO)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.20+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Start

### For Windows Users:
```bash
quick_start.bat
```

### For Linux/Mac Users:
```bash
chmod +x quick_start.sh
./quick_start.sh
```

### Manual Start:
```bash
# Install dependencies
pip install -r requirements_rl.txt

# Create sample datasets
python create_sample_dataset.py

# Launch web app
streamlit run app_rl_selector.py
```

---

## 📚 Full Documentation

**For complete documentation, see:** [README_RL_SELECTOR.md](README_RL_SELECTOR.md)

---

## 🎯 What is This?

This project uses **Reinforcement Learning (PPO)** to automatically select the best machine learning model for your dataset. Instead of manually trying different models, the RL agent learns from experience which models work best for different types of data.

### Key Features:
- 🤖 **Automatic Model Selection** - RL agent picks the best model
- 🔍 **Auto Task Detection** - Detects classification vs regression
- 🚀 **GPU Acceleration** - Fast training with CUDA support
- 🌐 **Web Interface** - Easy-to-use Streamlit app
- 📊 **Comprehensive Metrics** - Detailed performance analysis

---

## 📁 Project Files

### Core Files
- `app_rl_selector.py` - Streamlit web application
- `rl_model_selector_classification_regression.py` - Main RL selector
- `train_rl_model_selector.py` - Training script
- `test_rl_selector.py` - Testing/validation
- `create_sample_dataset.py` - Sample dataset generator

### Documentation
- `README_RL_SELECTOR.md` - **Full documentation** ⭐
- `CLEANUP_GUIDE.md` - Repository cleanup guide
- `requirements_rl.txt` - Dependencies

### Quick Start Scripts
- `quick_start.bat` - Windows quick start
- `quick_start.sh` - Linux/Mac quick start

---

## 🎬 How to Use

1. **Install**: Run quick start script or install manually
2. **Upload**: Upload your CSV dataset
3. **Select**: Choose target column
4. **Run**: Click "Run Model Selection"
5. **Train**: Train the recommended model
6. **Evaluate**: View performance metrics

---

## 🔧 Supported Models

### Classification (10 models)
- XGBoost, LightGBM, CatBoost (GPU-accelerated)
- Random Forest, Extra Trees, Gradient Boosting
- Logistic Regression, SVM, KNN, Naive Bayes

### Regression (11 models)
- XGBoost, LightGBM, CatBoost (GPU-accelerated)
- Random Forest, Extra Trees, Gradient Boosting
- Ridge, Lasso, ElasticNet, SVR, KNN

---

## 📊 Example Usage

```python
from rl_model_selector_classification_regression import RLModelSelector
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize selector
selector = RLModelSelector()
selector.load()

# Get recommendations
result = selector.select(df, target='target_column', top_k=3)
print(f"Top models: {result['models']}")
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU detected | Install CUDA toolkit and GPU-enabled PyTorch |
| No trained model | Run training: `python rl_model_selector_classification_regression.py --mode train` |
| Import errors | Install dependencies: `pip install -r requirements_rl.txt` |

**For detailed troubleshooting, see:** [README_RL_SELECTOR.md](README_RL_SELECTOR.md)

---

## 📝 License

MIT License - See LICENSE file

## 👨‍💻 Author

DataPilot AI Pro

---

**Get Started Now!** 🚀
- Run: `streamlit run app_rl_selector.py`
- Or use quick start scripts for automated setup