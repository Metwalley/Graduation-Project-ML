# تدريب موديل XGBoost للتوحد - نسخة نهائية للتسليم

import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, plot_importance

warnings.filterwarnings("ignore", category=UserWarning)

# ====== إعدادات الملفات ======
FILE_NAME = "Autism_Screening_Data_Combined.csv"   # حط المسار الصحيح لو مش في نفس الفولدر
MAX_AGE = 16   # بنشتغل على الأطفال لحد 16 سنة
SCALER_OUT = "age_scaler.joblib"
FEATURES_OUT = "autism_features.joblib"
XGB_OUT = "autism_xgb_model.joblib"
METRICS_OUT = "autism_xgb_metrics.joblib"

# ====== 1. تحميل وتهيئة الداتا ======
df = pd.read_csv(FILE_NAME)
df.columns = [c.strip() for c in df.columns]

# فلترة العمر (بالسنين)
df = df[df["Age"] <= MAX_AGE].copy()

# الأعمدة المستخدمة
feature_cols = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","Age","Sex","Jaundice","Family_ASD"]
target_col = "Class"

# تأكد من أسماء الأعمدة (لو عندك تهجئة مختلفة عدل هنا)
# تحويل القيم النصية لأرقام (map)
mappings = {
    "Sex": {"m": 1, "f": 0},
    "Jaundice": {"yes": 1, "no": 0},    # ملاحظة: في ملفك كانت "Jauundice"
    "Family_ASD": {"yes": 1, "no": 0},
    "Class": {"YES": 1, "NO": 0}
}
for col, mp in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mp)

# إسقاط أي صفوف ناقصة
df_final = df[feature_cols + [target_col]].dropna().copy()
print("DATA ROWS:", len(df_final))
print("Class counts:\n", df_final[target_col].value_counts())

# ====== 2. تحضير الميزات + scaling للعمر فقط ======
X = df_final[feature_cols].copy()
y = df_final[target_col].copy()

scaler = StandardScaler()
X["Age"] = scaler.fit_transform(X[["Age"]])
joblib.dump(scaler, SCALER_OUT)
joblib.dump(feature_cols, FEATURES_OUT)

# ====== 3. تقسيم Train/Test ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ====== 4. حساب scale_pos_weight (عشان imbalance) ======
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f} (neg={neg}, pos={pos})")

# ====== 5. ضبط موديل XGBoost مع early stopping ======
xgb = XGBClassifier(
    n_estimators=1000,         # كدة بنسمح للـ early stopping بتقليل الiterations
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

# بنستخدم validation set (هنا X_test) مع early stopping
xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

print("Best iteration (best_ntree_limit):", xgb.get_booster().best_iteration)

# ====== 6. تقييم الموديل ======
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {acc*100:.2f}% | ROC AUC: {roc_auc:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Autism", "Autism"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Age Distribution by Class
sns.kdeplot(data=df_final, x='Age', hue='Class', fill=True)
plt.title("Age Distribution by Class")
plt.show()

# Feature importance (gain)
plt.figure(figsize=(8,6))
plot_importance(xgb, max_num_features=15, importance_type="gain", xlabel="Gain")
plt.title("XGBoost Feature Importance (gain)")
plt.show()

# ====== 7. Cross-validation (اختياري: 5-fold أو 3-fold لو بتشغل محلي) ======
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
print("CV ROC AUC scores:", [round(s,4) for s in cv_scores])
print("CV mean ROC AUC:", round(cv_scores.mean(),4))

# ====== 8. حفظ الموديل والميتركس ======
joblib.dump(xgb, XGB_OUT)
joblib.dump({"accuracy": acc, "roc_auc": roc_auc, "cv_roc_auc": cv_scores.mean()}, METRICS_OUT)
print("Saved model and metrics:", XGB_OUT, METRICS_OUT)