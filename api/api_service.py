import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# ======= 0. إعدادات الملفات والتحميل =======
# لازم الملفات دي تكون موجودة في نفس مكان الكود
MODEL_FILE = 'autism_xgb_model.joblib' 
FEATURES_FILE = 'autism_features.joblib'
SCALER_FILE = 'age_scaler.joblib'
METRICS_FILE = 'autism_xgb_metrics.joblib'

# تحميل المكونات عند بدء تشغيل الـ API مرة واحدة
try:
    # تحميل الموديل والـ Features
    MODEL = joblib.load(MODEL_FILE)
    FEATURE_COLS = joblib.load(FEATURES_FILE)
    SCALER = joblib.load(SCALER_FILE)
    METRICS = joblib.load(METRICS_FILE)

    # حفظ الدقة لعرضها
    ACCURACY = METRICS.get('accuracy', 0.0) * 100
    ROC_AUC = METRICS.get('roc_auc', 0.0)

    print("STATUS: All ML components loaded successfully.")
    print(f"STATUS: Model Accuracy: {ACCURACY:.2f}% | AUC: {ROC_AUC:.4f}")

except FileNotFoundError as e:
    # لو فيه أي ملف ناقص، الـ API مش هيشتغل
    print(f"ERROR: Failed to load file: {e}")
    raise HTTPException(status_code=500, detail="ML model files are missing or corrupted.")

# تهيئة تطبيق FastAPI
app = FastAPI(
    title="Autism Diagnosis ML API",
    description="API for predicting Autism Spectrum Disorder using XGBoost model (Accuracy: {:.2f}%)".format(ACCURACY),
    version="1.0.0"
)

# ======= 1. تعريف شكل المدخلات (Pydantic Schema) =======
# ده الشكل اللي لازم فريق Spring يبعته في الـ JSON
class AutismFeatures(BaseModel):
    # أسئلة السلوك (Binary: 0 or 1)
    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int
    
    # الأعمدة الإضافية
    Age: int          # العمر (1-16)
    Sex: int          # 1=m, 0=f
    Jaundice: int     # 1=yes, 0=no
    Family_ASD: int   # 1=yes, 0=no

# ======= 2. بناء الـ Endpoint للتوقع =======

@app.post("/predict/autism")
def predict_autism(input_data: AutismFeatures):
    """
    Performs binary classification (Autism/No Autism) based on 14 input features.
    The response includes the diagnosis, probability, and model metrics.
    """
    
    # 1. تحويل المدخلات لـ Pandas DataFrame (المطلوبة للموديل)
    # بنحول الـ Pydantic object لـ dictionary
    data_dict = input_data.model_dump()
    df_input = pd.DataFrame([data_dict])
    
    # 2. تطبيق الـ Scaling على عمود Age (ضروري جداً)
    # بنعمل نسخة من عمود Age عشان منبوظش الداتا الأصلية
    age_value = df_input['Age'].values.reshape(-1, 1)
    # بنستخدم الـ SCALER اللي حفظناه
    df_input['Age'] = SCALER.transform(age_value)
    
    # 3. ترتيب الأعمدة (مهم جداً لدقة الموديل)
    try:
        df_input = df_input[FEATURE_COLS]
    except KeyError as e:
        # لو فريق الـ Backend بعت عمود ناقص أو غلط
        raise HTTPException(status_code=400, detail=f"Missing feature in input: {e}")

    # 4. التوقع والنسبة
    prediction = MODEL.predict(df_input)[0]
    # بناخد نسبة الاحتمال لـ (YES / 1)
    probability_asd = MODEL.predict_proba(df_input)[:, 1][0] 
    
    # 5. بناء رسالة النتيجة
    if prediction == 1:
        diagnosis_label = "Probable Autism Spectrum Disorder (ASD)"
        confidence = round(probability_asd * 100, 2)
        recommendation = "High probability detected. Seek immediate consultation with a specialist."
    else:
        diagnosis_label = "No strong signs of ASD"
        confidence = round((1 - probability_asd) * 100, 2)
        recommendation = "Low probability detected. Continue routine follow-up."
        
    return {
        "disorder": "Autism Spectrum Disorder",
        "diagnosis_label": diagnosis_label,
        "probability_percent": confidence,
        "diagnosis_code": int(prediction),
        "recommendation": recommendation,
        "model_performance": {
            "accuracy": round(ACCURACY, 2),
            "roc_auc": round(ROC_AUC, 4)
        }
    }

# ======= 3. تشغيل الـ API (Instruction) =======
if __name__ == "__main__":
    # لتشغيل الـ API، استخدم الأمر: uvicorn api_service:app --reload
    # (هنا بنحط الـ uvicorn.run بس عشان الكود يكون مكتمل)
    uvicorn.run(app, host="0.0.0.0", port=8000)