import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

# 1. تحميل الموديل
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "adhd_xgb_model_optimized.joblib"
# model_path = "adhd_xgb_model_optimized.joblib"

print(f"🔍 Loading model from: {model_path}...")

try:
    model = joblib.load(model_path)
    print("✅ Model Loaded Successfully!\n")
    
    # 2. استخراج أهمية المميزات (Feature Importance)
    # الموديل بيخزن أهمية كل عمود في variable اسمه feature_importances_
    importances = model.feature_importances_
    
    # أسماء الأعمدة (بنفس الترتيب اللي دربنا عليه)
    feature_names = [
        "Hyperactivity_Score", "Conduct_Problems", "Emotional_Problems", 
        "Peer_Problems", "Prosocial_Score", "Total_Difficulties", 
        "Externalizing_Score", "Internalizing_Score", "Impact_Score", 
        "APQ_Involvement", "APQ_Positive_Parenting", "APQ_Poor_Monitoring", 
        "APQ_Inconsistent_Discipline", "APQ_Corporal_Punishment", 
        "APQ_Other_Discipline", "Age", "Sex"
    ]
    
    # 3. ترتيبهم من الأهم للأقل أهمية
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("📊 === Model Brain Scan (Top Influencers) ===")
    print(feature_imp_df)
    print("===========================================\n")

    # 4. تحليل النتيجة
    top_feature = feature_imp_df.iloc[0]['Feature']
    print(f"💡 الموديل بيعتمد بشكل أساسي على: {top_feature}")
    
    if top_feature in ["Total_Difficulties", "Hyperactivity_Score", "Externalizing_Score", "Conduct_Problems"]:
        print("✅ اطمن! الموديل 'ذكي' وبيركز على الأعراض الأساسية للمرض.")
    else:
        print("⚠️ خد بالك! الموديل بيركز على حاجات فرعية، محتاجين مراجعة.")

except Exception as e:
    print(f"❌ Error: {e}")