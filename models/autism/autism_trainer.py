import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, plot_importance
from pathlib import Path
import os

# تجاهل التحذيرات غير المهمة ليكون الخرج نظيفاً
warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Autism_Screening_Data_Combined.csv"
FILE_NAME = str(DATA_PATH)
print(f"📂 Loading data from: {FILE_NAME}")

MAX_AGE = 16  # الفئة المستهدفة (أطفال)
FEATURES_OUT = "autism_features.joblib"  # أسماء الأعمدة عشان الـ API
XGB_OUT = "autism_xgb_model.joblib"      # الموديل النهائي
METRICS_OUT = "autism_xgb_metrics.joblib" # نتائج التقييم

def train_autism_model():
    print("🚀 Starting Autism Model Training...")

    # ====== 2. تحميل وتجهيز الداتا ======
    try:
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        print(f"❌ Error: The file '{FILE_NAME}' was not found.")
        return

    # تنظيف أسماء الأعمدة (إزالة المسافات)
    df.columns = [c.strip() for c in df.columns]

    # فلترة العمر (الأطفال فقط)
    print(f"📊 Filtering data for Age <= {MAX_AGE}...")
    df = df[df["Age"] <= MAX_AGE].copy()

    # تحديد الأعمدة المطلوبة
    feature_cols = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", 
                    "Age", "Sex", "Jaundice", "Family_ASD"]
    target_col = "Class"

    # === التحسين الأهم: تنظيف النصوص قبل التحويل (Robustness) ===
    # ده بيحمي الموديل لو الداتا جاية فيها مسافات أو حروف كبيرة
    text_cols = ["Sex", "Jaundice", "Family_ASD", "Class"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # خرائط التحويل (Mappings)
    mappings = {
        "Sex": {"m": 1, "f": 0},
        "Jaundice": {"yes": 1, "no": 0},
        "Family_ASD": {"yes": 1, "no": 0},
        "Class": {"yes": 1, "no": 0} # لاحظ: حولنا كله لـ small letters فوق
    }

    # تطبيق التحويل
    for col, mp in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mp)

    # التأكد من نظافة الداتا النهائية
    df_final = df[feature_cols + [target_col]].dropna().copy()
    
    print(f"✅ Data Ready: {len(df_final)} samples.")
    print(f"   - Class Distribution: {df_final[target_col].value_counts().to_dict()}")

    # ====== 3. التقسيم والتدريب ======
    X = df_final[feature_cols]
    y = df_final[target_col]

    # حفظ أسماء الـ Features عشان الـ API يطلبهم بنفس الترتيب
    joblib.dump(feature_cols, FEATURES_OUT)

    # تقسيم الداتا (Stratified عشان التوازن)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # حساب الوزن لمعالجة عدم التوازن (Imbalance Handling)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # تعريف الموديل (XGBoost)
    xgb = XGBClassifier(
        n_estimators=500,        # عدد الأشجار (كافٍ جداً مع التوقف المبكر)
        learning_rate=0.05,      # معدل تعلم هادئ لدقة أعلى
        max_depth=4,             # عمق متوسط لمنع الـ Overfitting
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    # التدريب
    print("🔄 Training XGBoost Model...")
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )

    # ====== 4. التقييم ======
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*30)
    print(f"🏆 Final Results:")
    print(f"   - Accuracy: {acc*100:.2f}%")
    print(f"   - ROC AUC:  {roc_auc:.4f}")
    print("="*30)

    # طباعة تقرير مفصل
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ====== 5. الحفظ والتصدير ======
    # لاحظ: مش بنحفظ Scaler خلاص لأن XGB مش محتاجه
    joblib.dump(xgb, XGB_OUT)
    
    metrics = {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    joblib.dump(metrics, METRICS_OUT)

    print(f"💾 Model Saved Successfully -> {XGB_OUT}")
    print(f"💾 Features List Saved -> {FEATURES_OUT}")

    # (اختياري) رسم الـ Feature Importance
    plt.figure(figsize=(10, 6))
    plot_importance(xgb, max_num_features=10, importance_type="gain", title="Top 10 Features (Gain)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_autism_model()