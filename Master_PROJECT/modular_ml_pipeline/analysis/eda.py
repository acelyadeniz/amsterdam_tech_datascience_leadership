# analysis/eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df, output_dir="results/eda_plots"):
    """
    Performs Exploratory Data Analysis (EDA) on the dataset and
    saves the generated plots to the specified directory.
    """
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    os.makedirs(output_dir, exist_ok=True)
    print(f"EDA plots will be saved to '{output_dir}'.")

    # 1. Descriptive Statistics
    print("\n[EDA] Step 1: Generating descriptive statistics...")
    descriptive_stats = df.describe().T
    descriptive_stats['skew'] = df.skew(numeric_only=True)
    print("Descriptive Statistics:")
    print(descriptive_stats)

    # 2. Target Variable Distribution
    print("\n[EDA] Step 2: Creating target variable distribution plot...")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='signal', data=df)
    plt.title('Class Balance: Signal vs. Background', fontsize=16)
    plt.xlabel('Class (0 = Background, 1 = SUSY Signal)')
    plt.ylabel('Event Count')
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=12)
    plt.savefig(os.path.join(output_dir, '1_target_distribution.pdf'), bbox_inches='tight')
    plt.close()

    # 3. Feature Distribution (Histograms)
    print("\n[EDA] Step 3: Creating feature distribution histograms...")
    df.drop('signal', axis=1).hist(bins=50, figsize=(22, 16), color='teal', edgecolor='black')
    plt.suptitle('Histograms of All 18 Features', y=1.0, fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, '2_feature_histograms.pdf'), bbox_inches='tight')
    plt.close()

    # 4. Correlation Matrix
    print("\n[EDA] Step 4: Creating correlation matrix heatmap...")
    plt.figure(figsize=(20, 16))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix Heatmap', fontsize=24)
    plt.savefig(os.path.join(output_dir, '3_correlation_heatmap.pdf'), bbox_inches='tight')
    plt.close()

    # Print highly correlated pairs
    high_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
    high_corr = high_corr[high_corr < 1].drop_duplicates()
    print("\nMost Highly Correlated Feature Pairs (abs > 0.7):")
    print(high_corr[high_corr > 0.7].head(10))


    print("--- Exploratory Data Analysis Complete ---\n")