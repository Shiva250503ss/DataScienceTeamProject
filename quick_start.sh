#!/bin/bash

# ============================================================================
# Quick Start Script for RL Model Selector
# ============================================================================

echo "========================================================================"
echo "  RL Model Selector - Quick Start"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found! Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"
echo ""

# Step 2: Create virtual environment (optional but recommended)
echo "Step 2: Setting up virtual environment..."
read -p "Create a virtual environment? (recommended) [y/N]: " create_venv

if [[ $create_venv == "y" ]] || [[ $create_venv == "Y" ]]; then
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Skipping creation."
    else
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi

    print_info "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    print_success "Virtual environment activated"
else
    print_warning "Skipping virtual environment creation"
fi
echo ""

# Step 3: Install dependencies
echo "Step 3: Installing dependencies..."
read -p "Install dependencies from requirements_rl.txt? [Y/n]: " install_deps

if [[ $install_deps != "n" ]] && [[ $install_deps != "N" ]]; then
    print_info "Installing packages..."
    pip install -r requirements_rl.txt

    # Ask about GPU support
    echo ""
    read -p "Do you have an NVIDIA GPU and want GPU acceleration? [y/N]: " has_gpu

    if [[ $has_gpu == "y" ]] || [[ $has_gpu == "Y" ]]; then
        echo "Select your CUDA version:"
        echo "  1) CUDA 11.8"
        echo "  2) CUDA 12.1"
        echo "  3) CPU only (no GPU)"
        read -p "Enter choice [1-3]: " cuda_choice

        case $cuda_choice in
            1)
                print_info "Installing PyTorch with CUDA 11.8..."
                pip install torch --index-url https://download.pytorch.org/whl/cu118
                ;;
            2)
                print_info "Installing PyTorch with CUDA 12.1..."
                pip install torch --index-url https://download.pytorch.org/whl/cu121
                ;;
            *)
                print_info "Installing PyTorch CPU version..."
                pip install torch
                ;;
        esac
    else
        print_info "Installing PyTorch CPU version..."
        pip install torch
    fi

    print_success "Dependencies installed"
else
    print_warning "Skipping dependency installation"
fi
echo ""

# Step 4: Check GPU availability
echo "Step 4: Checking GPU availability..."
$PYTHON_CMD -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null
echo ""

# Step 5: Create sample datasets
echo "Step 5: Creating sample datasets..."
read -p "Create sample datasets for testing? [Y/n]: " create_samples

if [[ $create_samples != "n" ]] && [[ $create_samples != "N" ]]; then
    print_info "Generating sample datasets..."
    $PYTHON_CMD create_sample_dataset.py
    print_success "Sample datasets created"
else
    print_warning "Skipping sample dataset creation"
fi
echo ""

# Step 6: Train RL models (optional)
echo "Step 6: Training RL models..."
print_info "Training RL models takes time (10-30 minutes depending on your hardware)"
read -p "Train RL models now? [y/N]: " train_now

if [[ $train_now == "y" ]] || [[ $train_now == "Y" ]]; then
    print_info "Training RL models... This may take a while."
    $PYTHON_CMD rl_model_selector_classification_regression.py --mode train --task both --n_datasets 150 --timesteps 50000
    print_success "RL models trained"
else
    print_warning "Skipping RL model training"
    print_info "You can train later with: python rl_model_selector_classification_regression.py --mode train --task both"
fi
echo ""

# Step 7: Launch Streamlit app
echo "Step 7: Launching Streamlit app..."
read -p "Launch the Streamlit web application now? [Y/n]: " launch_app

if [[ $launch_app != "n" ]] && [[ $launch_app != "N" ]]; then
    print_success "Launching Streamlit app..."
    print_info "The app will open in your browser at http://localhost:8501"
    print_info "Press Ctrl+C to stop the server"
    echo ""
    streamlit run app_rl_selector.py
else
    print_warning "Skipping app launch"
    echo ""
    print_info "To launch the app later, run: streamlit run app_rl_selector.py"
fi

echo ""
echo "========================================================================"
print_success "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:8501 in your browser"
echo "  2. Upload a CSV dataset"
echo "  3. Select target column"
echo "  4. Run model selection"
echo "  5. Train and evaluate!"
echo ""
echo "For more information, see README.md"
echo ""
