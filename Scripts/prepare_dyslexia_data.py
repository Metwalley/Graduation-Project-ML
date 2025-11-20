import pandas as pd
from pathlib import Path

# ====== 1. إعداد المسارات ======
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dyslexia"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "Dyslexia_Merged_Data.csv"

def prepare_dyslexia_data():
    print("🚀 Starting Dyslexia Data Preparation...")

    # ====== 2. تحميل الملفات (مع حل مشكلة الفاصلة المنقوطة) ======
    # بنستخدم sep=';' عشان الملفات مفصولة بـ ; مش ,
    try:
        df_tablet = pd.read_csv(RAW_DIR / "Dyt-tablet.csv", sep=';')
        df_desktop = pd.read_csv(RAW_DIR / "Dyt-desktop.csv", sep=';')
        print(f"📂 Loaded Tablet: {df_tablet.shape}, Desktop: {df_desktop.shape}")
    except FileNotFoundError:
        print("❌ Error: Files not found! Please put 'Dyt-tablet.csv' and 'Dyt-desktop.csv' in 'data/raw/dyslexia/'")
        return

    # ====== 3. الدمج (Concatenation) ======
    df_all = pd.concat([df_tablet, df_desktop], ignore_index=True)
    print(f"🔗 Merged Data Shape: {df_all.shape}")

    # ====== 4. اختيار الأعمدة النضيفة فقط ======
    # هنختار أعمدة الدقة (Accuracy) اللي مفهاش قيم ناقصة
    # + البيانات الأساسية (Gender, Nativelang, Age, Dyslexia)
    
    # قائمة الأعمدة المتاحة والمشتركة (تأكدنا منها بالتحليل)
    keep_cols = [
        'Gender', 'Nativelang', 'Age', 'Dyslexia',
        'Accuracy1', 'Accuracy2', 'Accuracy3', 'Accuracy4', 'Accuracy5', 
        'Accuracy6', 'Accuracy7', 'Accuracy8', 'Accuracy9', 'Accuracy10', 
        'Accuracy11', 'Accuracy12', 'Accuracy14', 'Accuracy15', 'Accuracy16', 
        'Accuracy17', 'Accuracy22', 'Accuracy23', 'Accuracy30'
    ]
    
    df_clean = df_all[keep_cols].copy()

    # ====== 5. التنظيف والتحويل (Encoding) ======
    print("🧹 Cleaning and Encoding...")
    
    # تحويل الهدف (Dyslexia)
    # Yes -> 1, No -> 0
    df_clean['Dyslexia'] = df_clean['Dyslexia'].map({'Yes': 1, 'No': 0})
    
    # تحويل النوع (Gender)
    # Male -> 0, Female -> 1
    df_clean['Gender'] = df_clean['Gender'].map({'Male': 0, 'Female': 1})
    
    # تحويل اللغة الأم (Nativelang)
    # Yes (لغته الأم) -> 1, No -> 0
    df_clean['Nativelang'] = df_clean['Nativelang'].map({'Yes': 1, 'No': 0})

    # التأكد من عدم وجود قيم ناقصة
    df_clean.dropna(inplace=True)
    
    # إعادة تسمية الهدف لـ Class (عشان نوحد مع باقي الموديلات)
    df_clean.rename(columns={'Dyslexia': 'Class'}, inplace=True)

    print(f"✅ Final Clean Rows: {len(df_clean)}")
    print(f"   - Class Distribution: {df_clean['Class'].value_counts().to_dict()}")

    # ====== 6. الحفظ ======
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_dyslexia_data()