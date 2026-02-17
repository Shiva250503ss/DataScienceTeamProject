"""
Create sample datasets for testing the RL Model Selector
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def create_classification_dataset(filename='sample_classification.csv'):
    """Create a sample classification dataset."""
    print(f"Creating classification dataset: {filename}")

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(20)])
    df['target'] = y

    # Save
    df.to_csv(filename, index=False)
    print(f"[OK] Saved {filename} - {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"     Classes: {df['target'].nunique()}")
    print(f"     Class distribution:\n{df['target'].value_counts()}")

def create_regression_dataset(filename='sample_regression.csv'):
    """Create a sample regression dataset."""
    print(f"\nCreating regression dataset: {filename}")

    # Generate synthetic data
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=10,
        random_state=42
    )

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(15)])
    df['target'] = y

    # Save
    df.to_csv(filename, index=False)
    print(f"[OK] Saved {filename} - {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"     Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
    print(f"     Target mean: {df['target'].mean():.2f}")

def create_real_world_examples():
    """Create more realistic example datasets."""

    # Example 1: Customer Churn (Classification)
    print("\nCreating customer churn dataset...")
    np.random.seed(42)

    n = 500
    df_churn = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(20, 120, n),
        'total_charges': np.random.uniform(100, 8000, n),
        'num_products': np.random.randint(1, 5, n),
        'has_support_ticket': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'contract_type': np.random.choice([0, 1, 2], n),  # 0: month-to-month, 1: 1-year, 2: 2-year
        'payment_method': np.random.choice([0, 1, 2, 3], n),
        'churn': np.random.choice([0, 1], n, p=[0.7, 0.3])  # Target
    })

    df_churn.to_csv('sample_customer_churn.csv', index=False)
    print(f"[OK] Saved sample_customer_churn.csv - {df_churn.shape}")

    # Example 2: House Price Prediction (Regression)
    print("\nCreating house price dataset...")

    n = 500
    df_house = pd.DataFrame({
        'square_feet': np.random.randint(800, 4000, n),
        'bedrooms': np.random.randint(1, 6, n),
        'bathrooms': np.random.randint(1, 4, n),
        'age_years': np.random.randint(0, 50, n),
        'lot_size': np.random.uniform(1000, 20000, n),
        'garage_spaces': np.random.randint(0, 4, n),
        'has_pool': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'has_basement': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'neighborhood_score': np.random.uniform(1, 10, n),
    })

    # Generate price based on features (realistic relationship)
    df_house['price'] = (
        df_house['square_feet'] * 150 +
        df_house['bedrooms'] * 10000 +
        df_house['bathrooms'] * 15000 -
        df_house['age_years'] * 500 +
        df_house['lot_size'] * 5 +
        df_house['garage_spaces'] * 8000 +
        df_house['has_pool'] * 20000 +
        df_house['has_basement'] * 15000 +
        df_house['neighborhood_score'] * 10000 +
        np.random.normal(0, 20000, n)  # Add noise
    )

    df_house.to_csv('sample_house_prices.csv', index=False)
    print(f"[OK] Saved sample_house_prices.csv - {df_house.shape}")

if __name__ == "__main__":
    print("="*70)
    print(" Creating Sample Datasets for RL Model Selector")
    print("="*70)

    # Create basic datasets
    create_classification_dataset()
    create_regression_dataset()

    # Create realistic examples
    create_real_world_examples()

    print("\n" + "="*70)
    print(" [OK] All sample datasets created successfully!")
    print("="*70)
    print("\nYou can now use these datasets to test the RL Model Selector:")
    print("  1. sample_classification.csv - Basic classification")
    print("  2. sample_regression.csv - Basic regression")
    print("  3. sample_customer_churn.csv - Customer churn prediction")
    print("  4. sample_house_prices.csv - House price prediction")
    print("\nRun: streamlit run app_rl_selector.py")
