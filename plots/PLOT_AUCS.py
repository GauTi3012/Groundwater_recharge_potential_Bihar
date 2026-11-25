# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:39:47 2026

@author: gauta
"""

# -*- coding: utf-8 -*-

import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# Define categorical layers for the OHE model
CATEGORICAL_LAYERS = ["GEOMORPHOLOGY", "LITHOLOGY", "LULC"]

def build_training_data(shp_path, raster_paths, label_col="Target"):
    """Extracts raster values at shapefile point locations."""
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
    
    # 1. Extract Training Data
    df_train = build_training_data(shp_path, my_rasters, label_col="GW_Status")
    
    # 2. Base Preprocessing (No OHE)
    X = df_train.drop(columns=['Target'])
    y = df_train['Target'].astype(int)
    
    # Use random_state=42 consistently to ensure all models are tested on the exact same holdout data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply SMOTE for imbalanced base data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # 3. OHE Preprocessing (For Model 5)
    X_ohe = X.copy()
    for col in CATEGORICAL_LAYERS:
        if col in X_ohe.columns:
            X_ohe[col] = X_ohe[col].astype(int).astype(str)
            
    X_encoded = pd.get_dummies(X_ohe, columns=[c for c in CATEGORICAL_LAYERS if c in X_ohe.columns])
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    
    X_train_enc_smote, y_train_enc_smote = smote.fit_resample(X_train_enc, y_train_enc)
    
    # 4. Initialize Dictionary to store ROC data
    roc_data = {}

    print("--- Training Models ---")
    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    probs_rf = rf.predict_proba(X_test)[:, 1]
    roc_data['Random Forest'] = roc_curve(y_test, probs_rf)
    
    # Model 2: Standard XGBoost
    xgb_base = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    xgb_base.fit(X_train, y_train)
    probs_xgb = xgb_base.predict_proba(X_test)[:, 1]
    roc_data['XGBoost'] = roc_curve(y_test, probs_xgb)

    # Model 3: Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    probs_gb = gb_model.predict_proba(X_test)[:, 1]
    roc_data['Gradient Boosting'] = roc_curve(y_test, probs_gb)

    # Model 4: XGBoost + SMOTE
    xgb_smote = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc')
    xgb_smote.fit(X_train_smote, y_train_smote)
    probs_xgb_smote = xgb_smote.predict_proba(X_test)[:, 1]
    roc_data['XGBoost + SMOTE'] = roc_curve(y_test, probs_xgb_smote)

    # Model 5: XGBoost + OHE + SMOTE
    xgb_enc_smote = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc')
    xgb_enc_smote.fit(X_train_enc_smote, y_train_enc_smote)
    probs_xgb_enc_smote = xgb_enc_smote.predict_proba(X_test_enc)[:, 1]
    roc_data['XGBoost + OHE + SMOTE'] = roc_curve(y_test_enc, probs_xgb_enc_smote)

    # 5. Plot Combined ROC Curve
    print("--- Generating Combined ROC Curve ---")
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'darkorange']
    
    for (model_name, (fpr, tpr, thresholds)), color in zip(roc_data.items(), colors):
        # Calculate AUC dynamically for the legend
        auc_score = roc_auc_score(y_test, probs_xgb_enc_smote) if "OHE" in model_name else roc_auc_score(y_test, [probs_rf, probs_xgb, probs_gb, probs_xgb_smote][list(roc_data.keys()).index(model_name)])
        # Re-calc AUC cleanly
        if "OHE" in model_name:
            auc = roc_auc_score(y_test_enc, probs_xgb_enc_smote)
        else:
             if model_name == 'Random Forest': auc = roc_auc_score(y_test, probs_rf)
             elif model_name == 'XGBoost (Standard)': auc = roc_auc_score(y_test, probs_xgb)
             elif model_name == 'Gradient Boosting': auc = roc_auc_score(y_test, probs_gb)
             elif model_name == 'XGBoost + SMOTE': auc = roc_auc_score(y_test, probs_xgb_smote)
        
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Combined ROC Curves for Groundwater Recharge Potential Models', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()