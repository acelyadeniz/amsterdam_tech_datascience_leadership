# data_handling/loader.py

import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Column names for the dataset
COLS = ["signal", "lepton 1 pT", "lepton 1 eta", "lepton 1 phi",
        "lepton 2 pT", "lepton 2 eta", "lepton 2 phi",
        "missing energy magnitude", "missing energy phi", "MET_rel",
        "axial MET", "M_R", "M_TR2", "R", "MT2", "S_R", "M_Delta_R",
        "dPhi_r_b", "cos(theta_r1)"]

LOW_LEVEL_FEATS = COLS[1:9]
HIGH_LEVEL_FEATS = COLS[9:]

def load_and_preprocess_data(n_rows=5_000_000, test_size=0.2, random_state=42):
    """
    Downloads, loads, preprocesses, splits, and scales the SUSY dataset.
    """
    print("--- Starting Data Loading and Preprocessing Step ---")
    try:
        # Download the dataset from KaggleHub
        path = kagglehub.dataset_download("janus137/supersymmetry-dataset")
        data_path = os.path.join(path, "supersymmetry_dataset.csv")
        print(f"Dataset downloaded/found at '{data_path}'.")
    except Exception as e:
        print(f"Could not download data from KaggleHub: {e}")
        raise FileNotFoundError("Dataset file not found. Please set the path manually.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Reading {n_rows} rows...")
    df_full = pd.read_csv(data_path, header=None, names=COLS, nrows=n_rows, skiprows=1)

    X = df_full.drop("signal", axis=1)
    y = df_full["signal"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data split into {1-test_size:.0%} training and {test_size:.0%} test sets.")
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    feature_sets = {
        "all": (X_train_scaled_df, X_test_scaled_df),
        "low_level": (X_train_scaled_df[LOW_LEVEL_FEATS], X_test_scaled_df[LOW_LEVEL_FEATS]),
        "high_level": (X_train_scaled_df[HIGH_LEVEL_FEATS], X_test_scaled_df[HIGH_LEVEL_FEATS])
    }
    
    print("--- Data Loading and Preprocessing Complete ---\n")
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, df_full, feature_sets