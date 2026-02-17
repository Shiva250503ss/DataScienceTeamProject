@echo off
REM ============================================================================
REM Quick Start Script for RL Model Selector (Windows)
REM ============================================================================

echo ========================================================================
echo   RL Model Selector - Quick Start
echo ========================================================================
echo.

REM Step 1: Check Python
echo Step 1: Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM Step 2: Virtual Environment
echo Step 2: Setting up virtual environment...
set /p CREATE_VENV="Create a virtual environment? (recommended) [Y/n]: "
if /i "%CREATE_VENV%"=="n" goto skip_venv

if exist "venv" (
    echo [WARNING] Virtual environment already exists. Skipping creation.
) else (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

:skip_venv
echo.

REM Step 3: Install Dependencies
echo Step 3: Installing dependencies...
set /p INSTALL_DEPS="Install dependencies from requirements_rl.txt? [Y/n]: "
if /i "%INSTALL_DEPS%"=="n" goto skip_deps

echo Installing packages...
pip install -r requirements_rl.txt

echo.
set /p HAS_GPU="Do you have an NVIDIA GPU and want GPU acceleration? [y/N]: "
if /i not "%HAS_GPU%"=="y" goto cpu_only

echo Select your CUDA version:
echo   1) CUDA 11.8
echo   2) CUDA 12.1
echo   3) CPU only (no GPU)
set /p CUDA_CHOICE="Enter choice [1-3]: "

if "%CUDA_CHOICE%"=="1" (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch --index-url https://download.pytorch.org/whl/cu118
) else if "%CUDA_CHOICE%"=="2" (
    echo Installing PyTorch with CUDA 12.1...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
) else (
    goto cpu_only
)
goto deps_done

:cpu_only
echo Installing PyTorch CPU version...
pip install torch

:deps_done
echo [OK] Dependencies installed

:skip_deps
echo.

REM Step 4: Check GPU
echo Step 4: Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>nul
echo.

REM Step 5: Create Sample Datasets
echo Step 5: Creating sample datasets...
set /p CREATE_SAMPLES="Create sample datasets for testing? [Y/n]: "
if /i "%CREATE_SAMPLES%"=="n" goto skip_samples

echo Generating sample datasets...
python create_sample_dataset.py
echo [OK] Sample datasets created

:skip_samples
echo.

REM Step 6: Train RL Models
echo Step 6: Training RL models...
echo [INFO] Training RL models takes time (10-30 minutes depending on your hardware)
set /p TRAIN_NOW="Train RL models now? [y/N]: "
if /i not "%TRAIN_NOW%"=="y" goto skip_training

echo Training RL models... This may take a while.
python rl_model_selector_classification_regression.py --mode train --task both --n_datasets 150 --timesteps 50000
echo [OK] RL models trained
goto training_done

:skip_training
echo [WARNING] Skipping RL model training
echo [INFO] You can train later with: python rl_model_selector_classification_regression.py --mode train --task both

:training_done
echo.

REM Step 7: Launch Streamlit
echo Step 7: Launching Streamlit app...
set /p LAUNCH_APP="Launch the Streamlit web application now? [Y/n]: "
if /i "%LAUNCH_APP%"=="n" goto skip_launch

echo [OK] Launching Streamlit app...
echo [INFO] The app will open in your browser at http://localhost:8501
echo [INFO] Press Ctrl+C to stop the server
echo.
streamlit run app_rl_selector.py
goto end

:skip_launch
echo [WARNING] Skipping app launch
echo.
echo [INFO] To launch the app later, run: streamlit run app_rl_selector.py

:end
echo.
echo ========================================================================
echo [OK] Setup Complete!
echo ========================================================================
echo.
echo Next steps:
echo   1. Open http://localhost:8501 in your browser
echo   2. Upload a CSV dataset
echo   3. Select target column
echo   4. Run model selection
echo   5. Train and evaluate!
echo.
echo For more information, see README.md
echo.
pause
