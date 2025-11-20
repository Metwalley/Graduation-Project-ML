from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Child Growth & Behvaior AI API", version="1.0")

# ====== 1. تحميل الموديلات عند البدء ======
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "ml_models"

models = {}

def load_model(name, filename):
    try:
        path = MODELS_DIR / filename
        return joblib.load(path)
    except Exception as e:
        print(f"⚠️ Error loading {name}: {e}")
        return None

@app.on_event("startup")
def load_all_models():
    # Autism
    models["autism_model"] = load_model("Autism Model", "autism_xgb_model.joblib")
    models["autism_features"] = load_model("Autism Features", "autism_features.joblib")
    
    # ADHD
    models["adhd_model"] = load_model("ADHD Model", "adhd_xgb_model_optimized.joblib")
    models["adhd_features"] = load_model("ADHD Features", "adhd_features.joblib")
    
    # Dyslexia
    models["dyslexia_model"] = load_model("Dyslexia Model", "dyslexia_rf_model.joblib")
    models["dyslexia_features"] = load_model("Dyslexia Features", "dyslexia_features.joblib")
    
    print("✅ All models loaded successfully!")

# ====== 2. تعريف شكل البيانات (Schemas) ======

# --- Autism Schema ---
class AutismInput(BaseModel):
    A1: str  # "Yes" or "No"
    A2: str
    A3: str
    A4: str
    A5: str
    A6: str
    A7: str
    A8: str
    A9: str
    A10: str
    Age: int
    Sex: str       # "m" or "f"
    Jaundice: str  # "yes" or "no"
    Family_ASD: str # "yes" or "no"

# --- ADHD Schema ---
class ADHDInput(BaseModel):
    # إجابات الأسئلة الـ 13 (0=لا، 1=أحياناً، 2=نعم)
    Q1: int # Hyperactivity
    Q2: int
    Q3: int
    Q4: int # Conduct
    Q5: int
    Q6: int # Emotional
    Q7: int # Peer
    Q8: int # Prosocial (Reverse)
    Q9: int # APQ Involvement
    Q10: int # APQ Positive
    Q11: int # APQ Poor Monitoring
    Q12: int # APQ Corporal Punishment
    Q13: int # APQ Inconsistent
    Age: int
    Sex: int # 0=Male, 1=Female (حسب تدريب الموديل)

# --- Dyslexia Schema ---
class DyslexiaInput(BaseModel):
    # قيم من 0.0 إلى 1.0 (حسب Slider أو نعم/لا في التطبيق)
    Language_vocab: float
    Memory: float
    Speed: float
    Visual_discrimination: float
    Audio_Discrimination: float
    Survey_Score: float

# ====== 3. الـ Endpoints (نقاط الاتصال) ======

@app.get("/")
def home():
    return {"message": "AI Service is Running! 🚀"}

# ---------------- AUTISM ENDPOINT ----------------
@app.post("/predict/autism")
def predict_autism(data: AutismInput):
    if not models["autism_model"]:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. تجهيز المدخلات (Mapping Logic)
    input_data = {
        "Age": data.Age,
        "Sex": 1 if data.Sex.lower() == 'm' else 0,
        "Jaundice": 1 if data.Jaundice.lower() == 'yes' else 0,
        "Family_ASD": 1 if data.Family_ASD.lower() == 'yes' else 0,
    }

    # تحويل الأسئلة (Mapping الأسئلة الإيجابية والسلبية)
    # الأسئلة من A1 لـ A9 (إجابة "نعم" = 0 سليم، "لا" = 1 خطر)
    questions_pos = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    for q in questions_pos:
        val = getattr(data, q)
        input_data[q] = 0 if val.lower() == "yes" else 1

    # السؤال A10 (إجابة "نعم" = 1 خطر، "لا" = 0 سليم)
    input_data["A10"] = 1 if data.A10.lower() == "yes" else 0

    # 2. الترتيب والـ DataFrame
    features_order = models["autism_features"]
    df_in = pd.DataFrame([input_data])[features_order]

    # 3. التوقع
    prediction = models["autism_model"].predict(df_in)[0]
    probability = models["autism_model"].predict_proba(df_in)[0][1]

    return {
        "result": "Autism Risk" if prediction == 1 else "Normal",
        "probability": round(float(probability) * 100, 2),
        "description": "High risk of Autism Traits" if prediction == 1 else "No significant traits detected"
    }

# ---------------- ADHD ENDPOINT ----------------
@app.post("/predict/adhd")
def predict_adhd(data: ADHDInput):
    if not models["adhd_model"]:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. حساب الـ Scores من الإجابات الخام
    # (تجميع الأسئلة عشان نكون الـ Features اللي الموديل عارفها)
    
    # Q1, Q2, Q3 -> Hyperactivity (Max 6)
    hyperactivity = data.Q1 + data.Q2 + data.Q3
    
    # Q4, Q5 -> Conduct Problems (Max 4)
    conduct = data.Q4 + data.Q5
    
    # Q6 -> Emotional (Max 2)
    emotional = data.Q6 
    
    # Q7 -> Peer Problems (Max 2)
    peer = data.Q7
    
    # Q8 -> Prosocial (Reverse Scoring: 2=0, 1=1, 0=2)
    # لأن السؤال إيجابي: هل يساعد الآخرين؟
    prosocial_raw = data.Q8
    prosocial = 2 - prosocial_raw
    
    # APQ Mappings (Direct)
    # لاحظ: الموديل متدرب على قيم كبيرة للـ APQ، هنضرب في فاكتور بسيط للتقريب
    # أو نعتمد المدخل المباشر لو هو Scale 1-5. 
    # هنا هنفترض الإجابة 0-2 وهنضربها في 2 عشان تقرب من رينج الداتا
    
    input_data = {
        "Conduct_Problems": conduct,
        "Total_Difficulties": conduct + emotional + peer + hyperactivity, # مجموع كلي
        "Emotional_Problems": emotional,
        "Externalizing_Score": conduct + hyperactivity,
        "Impact_Score": 0, # افتراضي (مش هنسأل عليه للتسهيل)
        "Hyperactivity_Score": hyperactivity,
        "Internalizing_Score": emotional + peer,
        "Peer_Problems": peer,
        "Prosocial_Score": prosocial,
        
        "APQ_Corporal_Punishment": data.Q12,
        "APQ_Inconsistent_Discipline": data.Q13,
        "APQ_Involvement": data.Q9, # إيجابي
        "APQ_Other_Discipline": 0, # افتراضي
        "APQ_Poor_Monitoring": data.Q11,
        "APQ_Positive_Parenting": data.Q10, # إيجابي
        
        "Age": data.Age,
        "Sex": data.Sex
    }

    # 2. الترتيب
    features_order = models["adhd_features"]
    df_in = pd.DataFrame([input_data])
    # ملء أي عمود ناقص بـ 0 (زي Impact_Score)
    for col in features_order:
        if col not in df_in.columns:
            df_in[col] = 0
    
    df_in = df_in[features_order]

    # 3. التوقع
    prediction = models["adhd_model"].predict(df_in)[0]
    probability = models["adhd_model"].predict_proba(df_in)[0][1]

    return {
        "result": "ADHD Risk" if prediction == 1 else "Normal",
        "probability": round(float(probability) * 100, 2)
    }

# ---------------- DYSLEXIA ENDPOINT ----------------
@app.post("/predict/dyslexia")
def predict_dyslexia(data: DyslexiaInput):
    if not models["dyslexia_model"]:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. تجهيز البيانات
    input_data = {
        "Language_vocab": data.Language_vocab,
        "Memory": data.Memory,
        "Speed": data.Speed,
        "Visual_discrimination": data.Visual_discrimination,
        "Audio_Discrimination": data.Audio_Discrimination,
        "Survey_Score": data.Survey_Score
    }

    features_order = models["dyslexia_features"]
    df_in = pd.DataFrame([input_data])[features_order]

    # 2. التوقع
    prediction = models["dyslexia_model"].predict(df_in)[0]
    # Random Forest Probability (Classes: 0, 1, 2)
    probs = models["dyslexia_model"].predict_proba(df_in)[0]
    
    # تفسير النتيجة (حسب الكود النهائي: 0=High, 1=Mod, 2=Low)
    labels_map = {0: "High Risk", 1: "Moderate Risk", 2: "Low Risk"}
    result_text = labels_map.get(prediction, "Unknown")

    return {
        "result": result_text,
        "risk_level": int(prediction), # 0 is worst
        "details": {
            "high_risk_prob": round(float(probs[0]) * 100, 2),
            "moderate_risk_prob": round(float(probs[1]) * 100, 2),
            "low_risk_prob": round(float(probs[2]) * 100, 2)
        }
    }