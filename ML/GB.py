# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:55:00 2026

@author: gauta
"""

import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier # Imported Gradient Boosting
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# ==========================================
# 1. Function Definitions
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

def build_training_data(shp_path, raster_paths, label_col="Target"):
    """Extracts raster values at shapefile point locations."""
    print("--- Extracting Training Data ---")
    points = gpd.read_file(shp_path)
    
    # Store coordinates for rasterio sampling
    coords = [(x, y) for x, y in zip(points.geometry.x, points.geometry.y)]
    
    # Initialize DataFrame with the target labels
    df_train = pd.DataFrame({'Target': points[label_col]})
    
    for path in raster_paths:
        name = os.path.basename(path).replace(".tif", "")
        with rasterio.open(path) as src:
            # Sample raster at given coordinates
            df_train[name] = [val[0] for val in src.sample(coords)]
            
    # Drop any points that fell on NoData pixels
    df_train.dropna(inplace=True)
    return df_train

def train_gb_model(df_train):
    """Trains a Gradient Boosting classifier and prints metrics, including ROC curve and Feature Importance."""
    print("--- Training Gradient Boosting Model ---")
    X = df_train.drop(columns=['Target'])
    y = df_train['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Tree-based models do not require scaled data. We can directly initialize the model.
    # n_estimators=100 and learning_rate=0.1 are standard starting parameters.
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    gb_model.fit(X_train, y_train)
    
    # Evaluation
    preds = gb_model.predict(X_test)
    probs = gb_model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, probs)
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # --- PLOT 1: Feature Importance ---
    importances = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    importances.sort_values(ascending=True).plot(kind='barh', color='mediumseagreen', edgecolor='black')
    plt.title('Top Feature Importances (Gradient Boosting)')
    plt.xlabel('Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # ----------------------------------------
    
    # --- PLOT 2: ROC CURVE ---
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f' GB (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Gradient Boosting')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    # ----------------------
    
    return gb_model

def predict_gwpi_map_chunked(model, raster_paths, reference_raster_path, output_path):
    """Predicts probabilities block-by-block to prevent MemoryErrors."""
    print("\n--- Generating GWPI Map in Chunks (Memory Safe) ---")
    
    # Get feature names in the exact order the model was trained on
    feature_names = [os.path.basename(p).replace(".tif", "") for p in raster_paths]
    
    with rasterio.open(reference_raster_path) as ref_src:
        meta = ref_src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=-9999)
        
        # Open all rasters simultaneously for reading
        srcs = [rasterio.open(p) for p in raster_paths]
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Iterate over internal blocks/windows of the raster
            for ji, window in ref_src.block_windows(1):
                
                block_data = {}
                # Read the exact same window from all 9 rasters
                for src, name in zip(srcs, feature_names):
                    arr = src.read(1, window=window)
                    # Convert to float32 immediately to save memory
                    arr_float = arr.astype(np.float32)
                    
                    if src.nodata is not None:
                        arr_float[arr_float == src.nodata] = np.nan
                        
                    block_data[name] = arr_float.flatten()
                
                # Create a temporary DataFrame for just this small chunk
                df_chunk = pd.DataFrame(block_data)
                
                # Identify valid pixels (pixels that are not NoData in any layer)
                valid_mask = df_chunk.notna().all(axis=1)
                
                # Initialize output block filled with NoData
                out_block = np.full(df_chunk.shape[0], -9999, dtype=np.float32)
                
                # Predict only on the valid pixels
                if valid_mask.any():
                    valid_pixels = df_chunk[valid_mask]
                    probs = model.predict_proba(valid_pixels)[:, 1]
                    out_block[valid_mask] = probs
                
                # Reshape back to the window's 2D dimensions and write to disk
                out_block_2d = out_block.reshape((window.height, window.width))
                dst.write(out_block_2d, 1, window=window)
        
        # Close all source rasters to free resources
        for src in srcs:
            src.close()
            
    print(f"Success! Full-resolution GWPI map saved to {output_path}")

# ==========================================
# 2. Execution Block
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
    
    shp_path = r"D:\project\shrinkhala\INDIA_CGWB\India_CGWB_CSV\India_CGWB_CSV\GWL\GWL_VAL2.shp"
    
    # Step 1: Extract training data
    df_train = build_training_data(shp_path, my_rasters, label_col="GW_Status")
    
    # Step 2: Train the Gradient Boosting Model
    gb_model = train_gb_model(df_train)
    
    # Step 3: Predict and export the final GeoTIFF block-by-block
    output_tif = r"D:\project\9factors\NEW_ML_tiff\GB.tif"
    reference_tif = my_rasters[0] 
    
    # Pass the list of raster paths directly to the chunking function
    predict_gwpi_map_chunked(gb_model, my_rasters, reference_tif, output_tif)   