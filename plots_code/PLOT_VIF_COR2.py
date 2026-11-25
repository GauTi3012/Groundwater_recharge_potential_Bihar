# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:35:40 2026

@author: gauta
"""

import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import normaltest, skew, kurtosis

# ==========================================
# 1. Calculation Functions
# ==========================================

def extract_raster_data(raster_paths, step_size=10):
    """Reads rasters, applies spatial thinning, and returns a clean DataFrame."""
    raster_data = {}
    print(f"Reading rasters and sampling every {step_size}th pixel...")
    
    for path in raster_paths:
        file_name = os.path.basename(path).replace(".tif", "") 
        with rasterio.open(path) as src:
            arr2d = src.read(1)[::step_size, ::step_size]
            arr = arr2d.flatten()
            
            nodata = src.nodata
            arr = arr.astype(float)
            if nodata is not None:
                arr[arr == nodata] = np.nan
                
            raster_data[file_name] = arr

    df = pd.DataFrame(raster_data)
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"Retained {len(df)} valid pixels (Dropped {initial_rows - len(df)} NoData pixels).\n")
    return df

def check_normality(df):
    """Calculates Skewness, Kurtosis, and D'Agostino's K-squared test for normality."""
    print("--- Normality Check ---")
    results = []
    
    for col in df.columns:
        s = skew(df[col])
        k = kurtosis(df[col])
        stat, p_value = normaltest(df[col])
        
        is_normal = "Yes" if p_value > 0.05 else "No"
        
        results.append({
            "Feature": col,
            "Skewness": round(s, 4),
            "Kurtosis": round(k, 4),
            "P-Value": p_value,
            "Normally Distributed?": is_normal
        })
        
    normality_df = pd.DataFrame(results)
    return normality_df

def calculate_vif(df):
    """Calculates VIF and Tolerance scores using the inverse correlation matrix."""
    print("--- VIF & Tolerance Calculation ---")
    
    corr_matrix = df.corr()
    
    try:
        inv_corr = np.linalg.inv(corr_matrix.values)
        vif_scores = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        print("Warning: Perfect multicollinearity detected. Using pseudo-inverse.")
        inv_corr = np.linalg.pinv(corr_matrix.values)
        vif_scores = np.diag(inv_corr)
        
    vif_results = pd.DataFrame({
        "Feature": df.columns,
        "VIF_Score": vif_scores,
        "Tolerance": 1 / vif_scores
    })
    
    return vif_results

# ==========================================
# 2. Plotting Functions
# ==========================================

def plot_all_boxplots(df):
    """Plots a grid of Box Plots for all raster features to visualize normality and outliers."""
    print("--- Plotting Box Plots for All Features ---")
    
    features = df.columns
    n_features = len(features)
    
    # Create a grid layout (3 columns wide)
    cols = 3
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    
    # Generate a list of distinct colors for the boxes
    colors = plt.cm.get_cmap('Pastel1')(np.linspace(0, 1, n_features))
    
    for i, col in enumerate(features):
        data = df[col].dropna()
        
        # Plot the individual box plot
        axes[i].boxplot(data, vert=False, patch_artist=True, 
                        boxprops=dict(facecolor=colors[i], color='black'),
                        medianprops=dict(color='red', linewidth=2))
        
        axes[i].set_title(f'Box Plot: {col}', fontsize=12)
        axes[i].set_yticks([]) 
        axes[i].set_xlabel('Value')
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle('Normality Analysis: Raster Feature Box Plots', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_vif_and_tolerance(vif_df):
    """Plots side-by-side bar charts for VIF and Tolerance with distinct colors per feature."""
    print("--- Plotting VIF and Tolerance ---")
    
    sorted_df = vif_df.sort_values(by="VIF_Score", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate distinct colors based on the number of features
    num_features = len(sorted_df)
    bar_colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, num_features))
    
    # 1. VIF Plot
    ax1.bar(sorted_df['Feature'], sorted_df['VIF_Score'], color=bar_colors, edgecolor='black')
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Multicollinearity Threshold (VIF=5)')
    ax1.set_title('Variance Inflation Factor (VIF)', fontsize=14)
    ax1.set_ylabel('VIF Score')
    ax1.set_xticklabels(sorted_df['Feature'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Tolerance Plot
    ax2.bar(sorted_df['Feature'], sorted_df['Tolerance'], color=bar_colors, edgecolor='black')
    ax2.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Low Tolerance Threshold (Tol=0.2)')
    ax2.set_title('Tolerance', fontsize=14)
    ax2.set_ylabel('Tolerance Score')
    ax2.set_xticklabels(sorted_df['Feature'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """Calculates and plots a correlation matrix heatmap using only Matplotlib."""
    print("--- Plotting Correlation Matrix ---")
    
    corr_matrix = df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)
    
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
    
    ax.set_title("Raster Variables Correlation Matrix", fontsize=16, pad=15)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. Execution Block
# ==========================================
if __name__ == "__main__":
    
    my_rasters = [
        r"D:\project\9factors\DRAINAGE_DENSITY.tif",
        r"D:\project\9factors\GEOMORPHOLOGY.tif",
        r"D:\project\9factors\LINEAMENT_DENSITY.tif",
        r"D:\project\9factors\LITHOLOGY.tif",
        r"D:\project\9factors\LULC.tif",
        r"D:\project\9factors\NDVI.tif",
        r"D:\project\9factors\RAINFALL.tif",
        r"D:\project\9factors\SLOPE.tif",
        r"D:\project\9factors\WZT.tif"
    ]
    
    # 1. Extract Data
    df = extract_raster_data(my_rasters, step_size=20)
    
    # 2. Check & Plot Normality (Grid of Box Plots)
    normality_df = check_normality(df)
    print(normality_df.to_string(index=False))
    print("\n")
    plot_all_boxplots(df)
    
    # 3. Check & Plot VIF/Tolerance (Distinct Colors)
    vif_df = calculate_vif(df)
    print(vif_df.sort_values(by="VIF_Score", ascending=False).to_string(index=False))
    print("\n")
    plot_vif_and_tolerance(vif_df)
    
    # 4. Plot Correlation Matrix
    plot_correlation_matrix(df)