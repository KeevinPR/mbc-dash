#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sample datasets for testing MBC algorithms

Creates multi-dimensional classification datasets with known structure
for validating the MBC-Dash application.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os


def generate_epilepsy_like_dataset(n_samples=500, missing_rate=0.1):
    """
    Generate epilepsy-like dataset similar to Benjumeda's work
    
    Classes: Y1 (Seizure Outcome), Y2 (Memory), Y3 (Language)  
    Features: Age, Lesion_Type, Seizure_Freq, etc.
    """
    np.random.seed(42)
    
    # Generate base features
    age = np.random.normal(35, 15, n_samples).astype(int)
    age = np.clip(age, 18, 70)
    
    lesion_type = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])  # 0=Hippocampal, 1=Neocortical, 2=Other
    seizure_freq = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])  # 0=Low, 1=Medium, 2=High
    duration = np.random.normal(15, 8, n_samples).astype(int)
    duration = np.clip(duration, 1, 40)
    
    # Generate correlated features
    mri_abnormal = np.random.binomial(1, 0.3 + 0.4 * (lesion_type > 0), n_samples)
    eeg_spikes = np.random.binomial(1, 0.2 + 0.5 * (seizure_freq > 1), n_samples)
    
    # Generate class variables with dependencies
    # Y1: Seizure Outcome (0=Poor, 1=Good)
    y1_prob = 0.6 - 0.3 * (lesion_type == 2) - 0.2 * (seizure_freq == 2) + 0.1 * (age < 30)
    y1 = np.random.binomial(1, np.clip(y1_prob, 0.1, 0.9), n_samples)
    
    # Y2: Memory Outcome (0=Declined, 1=Preserved) - depends on Y1 and lesion
    y2_prob = 0.5 + 0.3 * y1 - 0.2 * (lesion_type == 0) - 0.1 * (age > 50)
    y2 = np.random.binomial(1, np.clip(y2_prob, 0.1, 0.9), n_samples)
    
    # Y3: Language Outcome (0=Declined, 1=Preserved) - depends on lesion and age
    y3_prob = 0.7 - 0.4 * (lesion_type == 1) - 0.1 * (age > 45)
    y3 = np.random.binomial(1, np.clip(y3_prob, 0.1, 0.9), n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Y1_Seizure_Outcome': y1,
        'Y2_Memory': y2, 
        'Y3_Language': y3,
        'Age': age,
        'Lesion_Type': lesion_type,
        'Seizure_Frequency': seizure_freq,
        'Duration_Years': duration,
        'MRI_Abnormal': mri_abnormal,
        'EEG_Spikes': eeg_spikes
    })
    
    # Add missing values
    if missing_rate > 0:
        mask = np.random.random(df.shape) < missing_rate
        # Don't make class variables missing as often
        mask[:, :3] = mask[:, :3] * 0.3
        df = df.mask(mask)
    
    return df


def generate_synthetic_mbc_dataset(n_samples=300, n_classes=3, n_features=6):
    """
    Generate synthetic MBC dataset with known structure
    """
    np.random.seed(123)
    
    # Generate features using make_classification
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=42
    )
    
    # Discretize features
    X_discrete = np.zeros_like(X, dtype=int)
    for i in range(n_features):
        X_discrete[:, i] = pd.cut(X[:, i], bins=3, labels=[0, 1, 2])
    
    # Generate class variables with dependencies
    classes = np.zeros((n_samples, n_classes), dtype=int)
    
    # Class 1: depends on features 0, 1
    prob1 = 0.3 + 0.3 * (X_discrete[:, 0] > 0) + 0.2 * (X_discrete[:, 1] > 1)
    classes[:, 0] = np.random.binomial(1, prob1)
    
    # Class 2: depends on features 2, 3 and Class 1
    prob2 = 0.4 + 0.2 * (X_discrete[:, 2] > 0) + 0.1 * (X_discrete[:, 3] > 1) + 0.2 * classes[:, 0]
    classes[:, 1] = np.random.binomial(1, prob2)
    
    # Class 3: depends on features 4, 5
    prob3 = 0.5 + 0.3 * (X_discrete[:, 4] > 1) - 0.2 * (X_discrete[:, 5] > 0)
    classes[:, 2] = np.random.binomial(1, np.clip(prob3, 0.1, 0.9))
    
    # Create DataFrame
    data = np.hstack([classes, X_discrete])
    columns = [f'Class_{i+1}' for i in range(n_classes)] + [f'Feature_{i+1}' for i in range(n_features)]
    
    df = pd.DataFrame(data, columns=columns)
    return df


def generate_asia_extended_dataset(n_samples=400):
    """
    Generate extended version of classic Asia network for MBC testing
    """
    np.random.seed(456)
    
    # Original Asia variables (features)
    asia = np.random.binomial(1, 0.01, n_samples)  # Visit to Asia
    smoking = np.random.binomial(1, 0.5, n_samples)  # Smoking
    
    # Intermediate variables  
    tuberculosis = np.random.binomial(1, 0.05 + 0.95 * asia, n_samples)
    lung_cancer = np.random.binomial(1, 0.1 + 0.8 * smoking, n_samples)
    bronchitis = np.random.binomial(1, 0.45 + 0.45 * smoking, n_samples)
    
    either = (tuberculosis | lung_cancer).astype(int)
    
    # Symptoms (convert some to class variables for MBC)
    xray = np.random.binomial(1, 0.05 + 0.93 * either, n_samples)  # Class 1
    dyspnea = np.random.binomial(1, 0.1 + 0.7 * (either | bronchitis), n_samples)  # Class 2
    
    # Additional class variable
    severity = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        has_disease = int(tuberculosis[i] | lung_cancer[i])
        probs = [0.7 - 0.5 * has_disease, 0.2 + 0.3 * has_disease, 0.1 + 0.2 * has_disease]
        # Normalize probabilities
        probs = np.array(probs) / sum(probs)
        severity[i] = np.random.choice([0, 1, 2], p=probs)
    
    df = pd.DataFrame({
        'Xray_Abnormal': xray,      # Class 1
        'Dyspnea': dyspnea,         # Class 2  
        'Severity': severity,        # Class 3
        'Visit_Asia': asia,          # Feature 1
        'Smoking': smoking,          # Feature 2
        'Tuberculosis': tuberculosis, # Feature 3
        'Lung_Cancer': lung_cancer,   # Feature 4
        'Bronchitis': bronchitis,     # Feature 5
        'Either_Disease': either      # Feature 6
    })
    
    return df


def main():
    """Generate all sample datasets"""
    output_dir = "sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating sample datasets for MBC-Dash...")
    
    # Generate epilepsy-like dataset
    print("1. Generating epilepsy-like dataset...")
    epilepsy_df = generate_epilepsy_like_dataset(n_samples=500, missing_rate=0.15)
    epilepsy_df.to_csv(f"{output_dir}/epilepsy_synthetic.csv", index=False)
    print(f"   Saved: {output_dir}/epilepsy_synthetic.csv")
    print(f"   Shape: {epilepsy_df.shape}")
    print(f"   Classes: {list(epilepsy_df.columns[:3])}")
    print(f"   Features: {list(epilepsy_df.columns[3:])}")
    print(f"   Missing values: {epilepsy_df.isnull().sum().sum()}")
    
    # Generate synthetic MBC dataset
    print("\n2. Generating synthetic MBC dataset...")
    synthetic_df = generate_synthetic_mbc_dataset(n_samples=300, n_classes=3, n_features=6)
    synthetic_df.to_csv(f"{output_dir}/synthetic_mbc.csv", index=False)
    print(f"   Saved: {output_dir}/synthetic_mbc.csv")
    print(f"   Shape: {synthetic_df.shape}")
    print(f"   Classes: {list(synthetic_df.columns[:3])}")
    print(f"   Features: {list(synthetic_df.columns[3:])}")
    
    # Generate extended Asia dataset
    print("\n3. Generating extended Asia dataset...")
    asia_df = generate_asia_extended_dataset(n_samples=400)
    asia_df.to_csv(f"{output_dir}/asia_extended.csv", index=False)
    print(f"   Saved: {output_dir}/asia_extended.csv")
    print(f"   Shape: {asia_df.shape}")
    print(f"   Classes: {list(asia_df.columns[:3])}")
    print(f"   Features: {list(asia_df.columns[3:])}")
    
    print(f"\n✅ All datasets generated in '{output_dir}' directory")
    print("\nDataset descriptions:")
    print("- epilepsy_synthetic.csv: Medical dataset with 3 outcome classes and clinical features")
    print("- synthetic_mbc.csv: General MBC dataset with known dependencies")  
    print("- asia_extended.csv: Extended Asia network with multiple class variables")
    print("\nTo test MBC-Dash:")
    print("1. Run: python app.py")
    print("2. Upload one of these CSV files")
    print("3. Configure variables (first 3 columns as classes, rest as features)")
    print("4. Train algorithms and explore results!")


if __name__ == "__main__":
    main()
