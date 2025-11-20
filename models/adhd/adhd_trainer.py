import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ====== إعدادات المسارات ======
CURRENT_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = CURRENT_DIR.parent.parent       
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ADHD_Merged_Data.csv"
MODEL_OUT = CURRENT_DIR / "adhd_xgb_model_optimized.joblib"

def optimize_adhd_model():
    print("🚀 Starting Hyperparameter Tuning (Grid Search)...")
    print("☕ Go make some coffee, this might take a few minutes...")

    # تحميل الداتا
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = [
        "Conduct_Problems", "Total_Difficulties", "Emotional_Problems", 
        "Externalizing_Score", "Impact_Score", "Hyperactivity_Score", 
        "Internalizing_Score", "Peer_Problems", "Prosocial_Score",
        "APQ_Corporal_Punishment", "APQ_Inconsistent_Discipline", 
        "APQ_Involvement", "APQ_Other_Discipline", 
        "APQ_Poor_Monitoring", "APQ_Positive_Parenting",
        "Age", "Sex"
    ]
    target_col = "Class"
    
    X = df[feature_cols]
    y = df[target_col]

    # التقسيم
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # حساب الوزن
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # ====== إعداد الشبكة (The Grid) ======
    # هنجرب كل الاحتمالات دي
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300, 500],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1, 0.2] # معامل لمنع الـ Overfitting
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
        random_state=42
    )

    # البحث عن الأفضل (Grid Search)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc', # بنركز على الـ AUC أهم من الـ Accuracy
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n✅ Best Parameters Found:")
    print(grid_search.best_params_)

    # التقييم بالموديل المحسن
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*30)
    print(f"🏆 Optimized Results:")
    print(f"   - Accuracy: {acc*100:.2f}%")
    print(f"   - ROC AUC:  {roc_auc:.4f}")
    print("="*30)

    # حفظ الأفضل
    joblib.dump(best_model, MODEL_OUT)
    print(f"💾 Best Model Saved -> {MODEL_OUT}")

if __name__ == "__main__":
    optimize_adhd_model()