import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ====== إعدادات المسارات ======
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
# تأكد إن اسم الملف صح ومكانه صح
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "labeled_dysx.csv"

MODEL_OUT = CURRENT_DIR / "dyslexia_rf_model.joblib"
FEATURES_OUT = CURRENT_DIR / "dyslexia_features.joblib"
METRICS_OUT = CURRENT_DIR / "dyslexia_metrics.joblib"

def train_dyslexia_model():
    print("🚀 Starting Dyslexia Model Training (Final Approved Version)...")
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"📂 Data Loaded: {len(df)} samples")
    except FileNotFoundError:
        print(f"❌ Error: Data not found at {DATA_PATH}")
        return

    # ====== 1. تحديد الـ Features ======
    feature_cols = [
        'Language_vocab', 
        'Memory', 
        'Speed', 
        'Visual_discrimination', 
        'Audio_Discrimination', 
        'Survey_Score'
    ]
    target_col = 'Label'
    
    X = df[feature_cols]
    y = df[target_col]

    # تذكير بمعاني الـ Labels (حسب الـ Notebook الأصلي)
    print("ℹ️  Label Meanings: 0=High Risk (Dyslexia), 1=Moderate, 2=Low Risk (Normal)")
    print(f"📊 Class Distribution: {y.value_counts().to_dict()}")

    # حفظ أسماء الأعمدة
    joblib.dump(feature_cols, FEATURES_OUT)

    # ====== 2. التقسيم ======
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ====== 3. البحث عن الأفضل (GridSearch) ======
    # التحسين: استخدام f1_macro كما في الكود الأصلي للتعامل مع عدم التوازن
    print("🔄 Tuning Random Forest (Optimizing for F1 Macro)...")
    
    param_grid = {
        'n_estimators': [100, 200, 500],     # عدد الأشجار
        'max_depth': [None, 10, 20],         # عمق الشجرة
        'class_weight': ['balanced', None],  # موازنة الفئات
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro', # السر هنا! عشان يهتم بالفئات القليلة
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"✅ Best Params Found: {grid_search.best_params_}")

    # ====== 4. التقييم ======
    y_pred = best_rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*30)
    print(f"🏆 Final Results:")
    print(f"   - Accuracy: {acc*100:.2f}%")
    print(f"   - F1 Macro: {f1:.4f}")
    print("="*30)
    
    print(classification_report(y_test, y_pred))

    # ====== 5. الحفظ ======
    joblib.dump(best_rf, MODEL_OUT)
    joblib.dump({"accuracy": acc, "f1_macro": f1}, METRICS_OUT)
    print(f"💾 Model Saved: {MODEL_OUT}")
    
    # رسم الـ Confusion Matrix (بالترتيب المنطقي: High -> Moderate -> Low)
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]) # ترتيب الفئات يدوياً
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["High Risk", "Moderate", "Low Risk"])
    disp.plot(cmap="Reds_r") # أحمر للخطر، فاتح للأمان
    plt.title("Dyslexia Risk Prediction")
    plt.show()
    
    # رسم أهمية الأسئلة
    plt.figure(figsize=(10,6))
    feat_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title("Most Important Factors (Questions)")
    plt.show()

if __name__ == "__main__":
    train_dyslexia_model()