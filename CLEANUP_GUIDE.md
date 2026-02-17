# 🧹 Cleanup Guide - Keep Only RL Model Selector Files

This guide will help you clean up the repository to keep only the files related to the RL Model Selector.

## 📋 Files to KEEP

### Core RL Model Selector Files
- ✅ `rl_model_selector_classification_regression.py` - Main RL selector
- ✅ `train_rl_model_selector.py` - Training script
- ✅ `test_rl_selector.py` - Testing/validation
- ✅ `app_rl_selector.py` - Streamlit web application

### Configuration & Documentation
- ✅ `requirements_rl.txt` - Dependencies
- ✅ `README_RL_SELECTOR.md` - Main documentation
- ✅ `CLEANUP_GUIDE.md` - This file
- ✅ `LICENSE` - License file
- ✅ `.gitignore` - Git ignore rules

### Utilities
- ✅ `create_sample_dataset.py` - Sample dataset generator

### Directories
- ✅ `rl_model_selector_gpu/` - Trained models and logs
  - ✅ `rl_model_selector_gpu/models/` - PPO models
  - ✅ `rl_model_selector_gpu/logs/` - Training logs

### Git Files
- ✅ `.git/` - Git repository data
- ✅ `.github/` - GitHub workflows (if any)

## 🗑️ Files to REMOVE (Not related to RL Model Selector)

### Old Application Files
- ❌ `app_simple.py` - Old datapilot app
- ❌ `demo_simple.py` - Old demo

### Old Documentation
- ❌ `README.md` - Old readme (replace with README_RL_SELECTOR.md)
- ❌ `FLOW.md` - Old flow documentation
- ❌ `DataPilot_AI_Pro_Complete_Report.pdf` - Old report

### Old Requirements
- ❌ `requirements_simple.txt` - Old requirements (use requirements_rl.txt)

### Old Datapilot Directory
- ❌ `datapilot/` - Entire old datapilot module
  - ❌ `datapilot/agents/`
  - ❌ `datapilot/orchestrator.py`
  - ❌ `datapilot/__init__.py`

### IDE/Editor Files (Optional - can keep if you use them)
- ⚠️ `.vscode/` - VSCode settings (keep if you use VSCode)
- ⚠️ `.claude/` - Claude settings (keep if you use Claude)

## 🛠️ Manual Cleanup Steps

### Option 1: Manual Deletion (Safer)

1. **Delete Old Application Files:**
   ```bash
   rm app_simple.py demo_simple.py
   ```

2. **Delete Old Documentation:**
   ```bash
   rm FLOW.md DataPilot_AI_Pro_Complete_Report.pdf requirements_simple.txt
   ```

3. **Delete Old Datapilot Module:**
   ```bash
   rm -rf datapilot/
   ```

4. **Rename README:**
   ```bash
   mv README.md README_OLD.md  # Keep backup
   mv README_RL_SELECTOR.md README.md
   ```

### Option 2: Keep Everything Organized

If you want to keep old files for reference:

1. **Create Archive Directory:**
   ```bash
   mkdir archive
   ```

2. **Move Old Files:**
   ```bash
   mv app_simple.py demo_simple.py archive/
   mv datapilot/ archive/
   mv FLOW.md DataPilot_AI_Pro_Complete_Report.pdf archive/
   mv requirements_simple.txt archive/
   ```

3. **Update README:**
   ```bash
   mv README.md archive/README_OLD.md
   mv README_RL_SELECTOR.md README.md
   ```

## ✅ Final Clean Structure

After cleanup, your repository should look like this:

```
DataScienceTeamProject/
├── app_rl_selector.py
├── rl_model_selector_classification_regression.py
├── train_rl_model_selector.py
├── test_rl_selector.py
├── create_sample_dataset.py
├── requirements_rl.txt
├── README.md
├── LICENSE
├── .gitignore
├── rl_model_selector_gpu/
│   ├── models/
│   └── logs/
├── .git/
└── .vscode/ (optional)
```

## 🔍 Verification

After cleanup, verify everything works:

1. **Check dependencies:**
   ```bash
   pip install -r requirements_rl.txt
   ```

2. **Create sample datasets:**
   ```bash
   python create_sample_dataset.py
   ```

3. **Run the app:**
   ```bash
   streamlit run app_rl_selector.py
   ```

4. **Run tests (if models are trained):**
   ```bash
   python test_rl_selector.py
   ```

## ⚠️ Before You Delete

1. **Backup Important Data:**
   - Make sure you have backups of any important data
   - Consider creating a git tag: `git tag -a v1.0-datapilot -m "Before cleanup"`

2. **Check Dependencies:**
   - Some old files might be imported by others
   - Test after each deletion step

3. **Git History:**
   - Don't worry - git keeps history even after deletion
   - You can always recover files: `git checkout <commit> -- <file>`

## 🎯 Quick Cleanup Command

For experienced users who want to clean everything at once:

```bash
# Backup first!
git tag -a backup-$(date +%Y%m%d) -m "Backup before cleanup"

# Remove old files
rm -f app_simple.py demo_simple.py FLOW.md requirements_simple.txt DataPilot_AI_Pro_Complete_Report.pdf

# Remove old directory
rm -rf datapilot/

# Rename README
mv README.md README_OLD.md
cp README_RL_SELECTOR.md README.md

echo "✅ Cleanup complete!"
```

## 📝 Update Git

After cleanup, commit the changes:

```bash
git add -A
git commit -m "Cleanup: Keep only RL Model Selector files"
git push
```

---

**Need Help?** Check the README.md for full documentation and usage instructions.
