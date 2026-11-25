# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:47:31 2026

@author: gauta
"""

import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def build_training_data(shp_path, raster_paths, label_col="Target"):
    print("--- Extracting Training Data ---")
    points = gpd.read_file(shp_path)
    coords = [(x, y) for x, y in zip(points.geometry.x, points.geometry.y)]
    df_train = pd.DataFrame({'Target': points[label_col]})
    
    for path in raster_paths:
        name = os.path.basename(path).replace(".tif", "")
        with rasterio.open(path) as src:
            df_train[name] = [val[0] for val in src.sample(coords)]
            
    df_train.dropna(inplace=True)
    return df_train

def preprocess_and_train(df_train):
    print("--- Preprocessing & Training XGBoost ---")
    X = df_train.drop(columns=['Target'])
    y = df_train['Target'].astype(int)
    
    training_cols = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # SMOTE for class imbalance
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Train XGBoost (No OHE needed)
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc'
    )
    xgb.fit(X_train_smote, y_train_smote)
    
    # Evaluation & Metrics
    preds = xgb.predict(X_test)
    probs = xgb.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)
    
    print("\n" + classification_report(y_test, preds))
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # --- PLOT 1: Feature Importance ---
    importances = pd.Series(xgb.feature_importances_, index=training_cols).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    importances.sort_values(ascending=True).plot(kind='barh', color='mediumseagreen', edgecolor='black')
    plt.title('Top Feature Importances (XGBoost + SMOTE)')
    plt.xlabel('Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # --- PLOT 2: ROC AUC Curve ---
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost+SMOTE (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve - XGBoost + SMOTE')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
    return xgb, training_cols

def predict_gwpi_map_chunked(model, training_cols, raster_paths, reference_raster_path, output_path):
    print("\n--- Generating GWPI Map in Chunks ---")
    feature_names = [os.path.basename(p).replace(".tif", "") for p in raster_paths]
    
    with rasterio.open(reference_raster_path) as ref_src:
        meta = ref_src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=-9999)
        srcs = [rasterio.open(p) for p in raster_paths]
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for ji, window in ref_src.block_windows(1):
                block_data = {}
                for src, name in zip(srcs, feature_names):
                    arr = src.read(1, window=window)
                    arr_float = arr.astype(np.float32)
                    if src.nodata is not None:
                        arr_float[arr_float == src.nodata] = np.nan
                    block_data[name] = arr_float.flatten()
                
                df_chunk = pd.DataFrame(block_data)
                valid_mask = df_chunk.notna().all(axis=1)
                out_block = np.full(df_chunk.shape[0], -9999, dtype=np.float32)
                
                if valid_mask.any():
                    valid_pixels = df_chunk[valid_mask][training_cols]
                    probs = model.predict_proba(valid_pixels)[:, 1]
                    out_block[valid_mask] = probs
                
                out_block_2d = out_block.reshape((window.height, window.width))
                dst.write(out_block_2d, 1, window=window)
        
        for src in srcs:
            src.close()
            
    print(f"Success! GWPI map saved to {output_path}")

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
    
    # 1. Build Data
    df_train = build_training_data(shp_path, my_rasters, label_col="GW_Status")
    
    # 2. Train Model & Plot Metrics
    xgb_model, trained_columns = preprocess_and_train(df_train)
    
    # 3. Export GWPI Map
    reference_tif = my_rasters[0] 
    output_tif = r"D:\project\9factors\NEW_ML_tiff\XG_SMOTE.tif"
    
    predict_gwpi_map_chunked(xgb_model, trained_columns, my_rasters, reference_tif, output_tif)