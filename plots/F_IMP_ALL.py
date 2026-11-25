# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:02:19 2026

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
    
    # 2. Base Preprocessing
    X = df_train.drop(columns=['Target'])
    y = df_train['Target'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # 3. OHE Preprocessing
    X_ohe = X.copy()
    for col in CATEGORICAL_LAYERS:
        if col in X_ohe.columns:
            X_ohe[col] = X_ohe[col].astype(int).astype(str)
            
    X_encoded = pd.get_dummies(X_ohe, columns=[c for c in CATEGORICAL_LAYERS if c in X_ohe.columns])
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    X_train_enc_smote, y_train_enc_smote = smote.fit_resample(X_train_enc, y_train_enc)
    
    # Dictionary to store feature importances
    importance_dict = {}

    print("--- Training Models for Feature Importance ---")
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importance_dict['Random Forest'] = pd.Series(rf.feature_importances_, index=X.columns)
    
    # Model 2: Standard XGBoost
    xgb_base = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    xgb_base.fit(X_train, y_train)
    importance_dict['XGBoost'] = pd.Series(xgb_base.feature_importances_, index=X.columns)

    # Model 3: Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    importance_dict['Gradient Boosting'] = pd.Series(gb_model.feature_importances_, index=X.columns)

    # Model 4: XGBoost + SMOTE
    xgb_smote = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc')
    xgb_smote.fit(X_train_smote, y_train_smote)
    importance_dict['XGBoost + SMOTE'] = pd.Series(xgb_smote.feature_importances_, index=X.columns)

    # Model 5: XGBoost + OHE + SMOTE
    xgb_enc_smote = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc')
    xgb_enc_smote.fit(X_train_enc_smote, y_train_enc_smote)
    
    # Aggregate the OHE feature importances back to their base variables
    raw_ohe_importance = pd.Series(xgb_enc_smote.feature_importances_, index=X_encoded.columns)
    ohe_agg_importance = {}
    
    for base_col in X.columns:
        if base_col in CATEGORICAL_LAYERS:
            # Sum all dummy columns related to this categorical feature
            rel_cols = [c for c in X_encoded.columns if c.startswith(base_col + '_')]
            ohe_agg_importance[base_col] = raw_ohe_importance[rel_cols].sum()
        else:
            ohe_agg_importance[base_col] = raw_ohe_importance[base_col]
            
    importance_dict['XGBoost + OHE + SMOTE'] = pd.Series(ohe_agg_importance)

    # 4. Generate Combined Plot
    print("\n--- Generating Combined Feature Importance Plot ---")
    
    # Create DataFrame and normalize so all models add up to 100% for fair visual comparison
    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.div(importance_df.sum(axis=0), axis=1) * 100
    
    # Sort by the average importance across all models for better visual hierarchy
    importance_df['Average'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values(by='Average', ascending=True).drop(columns=['Average'])

    # Plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax = importance_df.plot(kind='barh', figsize=(12, 10), width=0.8, color=colors, edgecolor='black')
    
    plt.title('Combined Relative Feature Importances Across All Models', fontsize=16, pad=15)
    plt.xlabel('Relative Importance (%)', fontsize=12)
    plt.ylabel('Predictor Variables', fontsize=12)
    plt.legend(title='Models', loc='lower right', fontsize=11, title_fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()