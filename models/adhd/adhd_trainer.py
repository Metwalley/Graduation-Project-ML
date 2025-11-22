import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ADHD_Merged_Data.csv"

# Output Files
MODEL_OUT = CURRENT_DIR / "adhd_xgb_model_optimized.joblib"
FEATURES_OUT = CURRENT_DIR / "adhd_features.joblib"

def train_adhd_model():
    print("🚀 Starting ADHD Training (Fact-Based Normalization)...")
    
    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"📂 Data Loaded Successfully: {len(df)} rows")
    except FileNotFoundError:
        print(f"❌ Error: Data not found at {DATA_PATH}")
        return
    
    # ==========================================
    # 3. FACT-BASED NORMALIZATION
    # ==========================================
    # Max values observed in the dataset (HBN) to normalize inputs to [0.0 - 1.0]
    facts_max = {
        "Hyperactivity_Score": 10.0,
        "Conduct_Problems": 10.0,
        "Emotional_Problems": 10.0,
        "Peer_Problems": 9.0,
        "Prosocial_Score": 10.0,
        "Total_Difficulties": 32.0,
        "Externalizing_Score": 20.0,
        "Internalizing_Score": 16.0,
        "Impact_Score": 10.0,
        "APQ_Involvement": 50.0,
        "APQ_Positive_Parenting": 30.0,
        "APQ_Poor_Monitoring": 37.0,
        "APQ_Inconsistent_Discipline": 28.0,
        "APQ_Corporal_Punishment": 12.0,
        "APQ_Other_Discipline": 27.0
    }
    
    feature_cols = list(facts_max.keys()) + ["Age", "Sex"]
    target_col = "Class"
    
    # Apply Normalization
    X = df[feature_cols].copy()
    y = df[target_col]

    print("⚖️ Normalizing features...")
    for col, max_val in facts_max.items():
        if col in X.columns:
            # Clip outliers to max_val, then divide to get 0-1 range
            X[col] = X[col].clip(upper=max_val) / max_val

    # Save Feature Names for API Consistency
    joblib.dump(list(X.columns), FEATURES_OUT)
    print(f"📝 Features list saved to {FEATURES_OUT}")

    # ==========================================
    # 4. TRAIN/TEST SPLIT
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight for imbalance handling
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # ==========================================
    # 5. MODEL TRAINING (XGBoost + GridSearch)
    # ==========================================
    print("🔄 Tuning XGBoost Hyperparameters...")
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )

    # Hyperparameter Grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"✅ Best Params: {grid.best_params_}")

    # ==========================================
    # 6. EVALUATION
    # ==========================================
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*30)
    print(f"🏆 Final Results (Normalized Model):")
    print(f"   - Accuracy: {acc*100:.2f}%")
    print(f"   - ROC AUC:  {roc:.4f}")
    print("="*30)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ==========================================
    # 7. SAVE MODEL
    # ==========================================
    joblib.dump(best_model, MODEL_OUT)
    print(f"💾 Model saved successfully to: {MODEL_OUT}")

if __name__ == "__main__":
    train_adhd_model()