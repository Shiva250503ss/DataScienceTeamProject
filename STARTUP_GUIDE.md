# Quick Startup Guide

## ⚠️ Common Issues Fixed

### Issue 1: Installing Dependencies

**❌ WRONG:**
```bash
pip install requirements_simple.txt
```

**✅ CORRECT:**
```bash
pip install -r requirements_simple.txt
```

The `-r` flag tells pip to read from a requirements file.

---

## 🚀 Complete Setup Instructions

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd C:\Users\shiva\OneDrive\Documents\Github\DataScienceTeamProject

# Install all required packages
pip install -r requirements_simple.txt

# OR install individually if needed
pip install pandas numpy scikit-learn streamlit matplotlib seaborn scipy
```

### Step 2: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, streamlit; print('All packages installed!')"
```

If you see "All packages installed!", you're good to go!

---

## 🎯 Running the Applications

### Option A: Web Interface (Streamlit)

```bash
streamlit run profiler_cleaner_webapp.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.202.232.97:8501
```

Open your browser and go to: **http://localhost:8501**

**If you see errors:**
- The app has been updated to handle sklearn issues on Windows
- Try the "Generate Synthetic" button instead of Iris/Wine
- Or upload your own CSV file

---

### Option B: Command Line Demo

```bash
# Run with synthetic data
python profiler_cleaner_demo.py

# Run with Iris dataset
python profiler_cleaner_demo.py iris

# Run with your own CSV
python profiler_cleaner_demo.py your_data.csv
```

---

## 🐛 Troubleshooting

### Error: "RecursionError" with sklearn

**Cause:** Known issue with sklearn on some Windows systems

**Solution:** The web app has been updated with fallback:
- It will automatically use synthetic data if sklearn fails
- You'll see a warning: "Using synthetic iris-like dataset due to sklearn issue"
- This is normal and the demo will work fine

### Error: "No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit
```

### Error: "No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn
```

### Error: Package version conflicts

**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then install requirements
pip install -r requirements_simple.txt
```

---

## ✅ Quick Test

### Test the Command Line Demo:

```bash
python profiler_cleaner_demo.py
```

**Expected output:**
```
>> Available dataset options:
   - 'synthetic' : Create a synthetic dataset with various characteristics
   - 'titanic'   : Load Titanic dataset from seaborn
   - 'iris'      : Load Iris dataset from sklearn
   - 'wine'      : Load Wine dataset from sklearn
   - '<file.csv>': Load your own CSV file

   Using: synthetic

================================================================================
 DataPilot: Profiler & Cleaner Demo
================================================================================

>> Creating synthetic dataset with:
   - 500 samples
   - 15 numeric features
   - 3 categorical features
   ...
```

If you see this, it's working! ✅

---

### Test the Web App:

```bash
streamlit run profiler_cleaner_webapp.py
```

Then in your browser (http://localhost:8501):
1. Click "🎲 Generate Synthetic" button
2. Click "🚀 Analyze & Clean Dataset"
3. See the results!

---

## 📝 What Each File Does

| File | Purpose | How to Run |
|------|---------|------------|
| `profiler_cleaner_demo.py` | Command-line demo | `python profiler_cleaner_demo.py` |
| `profiler_cleaner_webapp.py` | Web interface | `streamlit run profiler_cleaner_webapp.py` |
| `test_both_tasks.py` | Test classification & regression | `python test_both_tasks.py` |
| `datapilot/agents/profiler.py` | Core profiler code | (imported by demos) |
| `datapilot/agents/cleaner.py` | Core cleaner code | (imported by demos) |

---

## 💡 Tips

1. **First time?** Start with the web app - it's visual and easier
2. **Got CSV data?** Upload it directly in the web app
3. **Want to automate?** Use the command-line demo
4. **Learning how it works?** Read `COMPLETE_AGENT_EXPLANATION.md`

---

## 🆘 Still Having Issues?

### Check Python Version:
```bash
python --version
```

Should be Python 3.7 or higher (you have 3.10, which is good!)

### Check if packages are installed:
```bash
pip list | findstr "pandas numpy scikit streamlit"
```

### Reinstall everything:
```bash
pip uninstall -y pandas numpy scikit-learn streamlit matplotlib seaborn scipy
pip install -r requirements_simple.txt
```

---

## ✅ Success Checklist

- [ ] Installed dependencies with `pip install -r requirements_simple.txt`
- [ ] Verified installation with `python -c "import pandas, numpy, sklearn, streamlit; print('OK')"`
- [ ] Ran command-line demo: `python profiler_cleaner_demo.py`
- [ ] Ran web app: `streamlit run profiler_cleaner_webapp.py`
- [ ] Uploaded a CSV or used Generate Synthetic button
- [ ] Saw quality score, meta-features, and cleaning results

If all boxes are checked, you're ready to use the system! 🎉

---

## 🚀 Next Steps

1. **Try with your own data**: Upload a CSV in the web app
2. **Read the documentation**: `COMPLETE_AGENT_EXPLANATION.md`
3. **Explore the features**: See all 32 meta-features
4. **Download results**: Get cleaned data and reports

**Happy data profiling and cleaning!** 📊
