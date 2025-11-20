import pandas as pd
import os
from pathlib import Path

# ====== 1. إعدادات المسارات (Dynamic Paths) ======
# المسار الحالي: .../Graduation-Project-ML/Scripts/
CURRENT_DIR = Path(__file__).resolve().parent

# الرجوع للرئيسية: .../Graduation-Project-ML/
PROJECT_ROOT = CURRENT_DIR.parent

# مسار الداتا الخام (Raw Data) - داخل فولدر TRAIN_NEW
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "widsdatathon2025" / "TRAIN_NEW"

# مسار حفظ الداتا المعالجة (Processed)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True) # إنشاء الفولدر لو مش موجود

# أسماء الملفات
FILE_SOLUTIONS = DATA_DIR / "TRAINING_SOLUTIONS.xlsx"
FILE_QUANTITATIVE = DATA_DIR / "TRAIN_QUANTITATIVE_METADATA_new.xlsx"
# FILE_CATEGORICAL = DATA_DIR / "TRAIN_CATEGORICAL_METADATA_new.xlsx" # مش محتاجينه دلوقتي

# ملف الخرج النهائي
OUTPUT_FILE = PROCESSED_DIR / "ADHD_Merged_Data.csv"

def merge_adhd_data():
    print("🚀 Starting Data Merge Process...")
    
    # ====== 2. تحميل الملفات ======
    try:
        print(f"📂 Loading files from: {DATA_DIR}")
        # استخدام engine='openpyxl' عشان ملفات الـ Excel الحديثة
        df_target = pd.read_excel(FILE_SOLUTIONS, engine='openpyxl')
        df_quant = pd.read_excel(FILE_QUANTITATIVE, engine='openpyxl')
        
        print(f"   - Targets: {df_target.shape}")
        print(f"   - Quantitative: {df_quant.shape}")

    except FileNotFoundError as e:
        print(f"❌ Error: File not found! Check the path.\nDetails: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # ====== 3. دمج الملفات (Merging) ======
    print("🔗 Merging data (Inner Join on participant_id)...")
    df_merged = pd.merge(df_quant, df_target, on="participant_id", how="inner")

    # ====== 4. اختيار وتنظيف الأعمدة (Feature Selection & Renaming) ======
    
    # خريطة تغيير الأسماء (لأسماء واضحة ومقروءة)
    rename_mapping = {
        # بيانات الطفل (SDQ)
        "SDQ_SDQ_Conduct_Problems": "Conduct_Problems",
        "SDQ_SDQ_Difficulties_Total": "Total_Difficulties",
        "SDQ_SDQ_Emotional_Problems": "Emotional_Problems",
        "SDQ_SDQ_Externalizing": "Externalizing_Score",
        "SDQ_SDQ_Generating_Impact": "Impact_Score",
        "SDQ_SDQ_Hyperactivity": "Hyperactivity_Score",
        "SDQ_SDQ_Internalizing": "Internalizing_Score",
        "SDQ_SDQ_Peer_Problems": "Peer_Problems",
        "SDQ_SDQ_Prosocial": "Prosocial_Score",
        
        # بيانات الأهل (APQ)
        "APQ_P_APQ_P_CP": "APQ_Corporal_Punishment",
        "APQ_P_APQ_P_ID": "APQ_Inconsistent_Discipline",
        "APQ_P_APQ_P_INV": "APQ_Involvement",
        "APQ_P_APQ_P_OPD": "APQ_Other_Discipline",
        "APQ_P_APQ_P_PM": "APQ_Poor_Monitoring",
        "APQ_P_APQ_P_PP": "APQ_Positive_Parenting",

        # البيانات الأساسية
        "MRI_Track_Age_at_Scan": "Age",
        "Sex_F": "Sex",         # 1=Female, 0=Male
        "ADHD_Outcome": "Class" # 1=ADHD, 0=No
    }

    # تصفية الداتا بالأعمدة اللي محتاجينها بس
    # (بنختار المفاتيح بتاعة الـ mapping + participant_id)
    selected_cols = ["participant_id"] + list(rename_mapping.keys())
    
    # التأكد إن الأعمدة موجودة قبل الاختيار
    available_cols = [c for c in selected_cols if c in df_merged.columns]
    df_final = df_merged[available_cols].copy()

    # تغيير الأسماء
    df_final.rename(columns=rename_mapping, inplace=True)

    # ====== 5. التنظيف النهائي (Cleaning) ======
    
    print(f"🧹 Cleaning missing data (Rows before: {len(df_final)})...")
    
    # [هام] حذف القيم الفارغة أولاً
    df_final.dropna(inplace=True)
    
    # [هام] تحويل العمر لرقم صحيح بعد الحذف
    if "Age" in df_final.columns:
        df_final["Age"] = df_final["Age"].round().astype(int)

    print(f"✅ Rows after cleaning: {len(df_final)}")

    # ====== 6. الحفظ (Saving) ======
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("="*30)
    print(f"💾 SAVED SUCCESSFULLY:\n   {OUTPUT_FILE}")
    print("="*30)
    
    # عرض عينة للتأكد
    print("\nSample Data:")
    print(df_final[["participant_id", "Class", "Age", "Hyperactivity_Score"]].head())
    
    print("\nClass Distribution:")
    print(df_final["Class"].value_counts())

if __name__ == "__main__":
    merge_adhd_data()