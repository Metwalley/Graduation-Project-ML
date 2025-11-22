import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

# ==========================================
# 1. APP CONFIGURATION & MODEL LOADING
# ==========================================
app = FastAPI(title="Child Psychological Assessment API", version="FINAL_ULTIMATE_V2")

# --- DYNAMIC PATH CONFIGURATION ---
# This ensures the code finds files regardless of where main.py is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")

# Define full paths to the files
AUTISM_MODEL_PATH = os.path.join(MODELS_DIR, "autism_xgb_model.joblib")
AUTISM_FEATURES_PATH = os.path.join(MODELS_DIR, "autism_features.joblib")
ADHD_MODEL_PATH = os.path.join(MODELS_DIR, "adhd_xgb_model_optimized.joblib")
DYSLEXIA_MODEL_PATH = os.path.join(MODELS_DIR, "dyslexia_rf_model.joblib")

# Global variables to store models
models = {}

def load_models():
    """Loads all ML models and feature lists into memory on startup."""
    print(f"⏳ Loading AI Models from: {MODELS_DIR} ...")
    try:
        # 1. Load Autism Model & Features
        models["autism"] = joblib.load(AUTISM_MODEL_PATH)
        models["autism_features"] = joblib.load(AUTISM_FEATURES_PATH)
        
        # 2. Load ADHD Model (Normalized)
        models["adhd"] = joblib.load(ADHD_MODEL_PATH)
        # Explicit feature order matching the Normalized Trainer
        models["adhd_features"] = [
            "Hyperactivity_Score", "Conduct_Problems", "Emotional_Problems", 
            "Peer_Problems", "Prosocial_Score", "Total_Difficulties", 
            "Externalizing_Score", "Internalizing_Score", "Impact_Score", 
            "APQ_Involvement", "APQ_Positive_Parenting", "APQ_Poor_Monitoring", 
            "APQ_Inconsistent_Discipline", "APQ_Corporal_Punishment", 
            "APQ_Other_Discipline", "Age", "Sex"
        ]
        
        # 3. Load Dyslexia Model
        models["dyslexia"] = joblib.load(DYSLEXIA_MODEL_PATH)
        
        print("✅ All models loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found. Please check 'ml_models' folder.\nDetails: {e}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load models. Details: {e}")

# Load models immediately
load_models()

# ==========================================
# 2. DATA SCHEMAS (Pydantic)
# ==========================================

class AnswerItem(BaseModel):
    q_id: int       # Question ID
    answer: str     # "yes", "no", "sometimes"

class AssessmentRequest(BaseModel):
    test_type: str  # "autism", "adhd", "dyslexia"
    age: float
    sex: str        # "m" or "f"
    jaundice: Optional[str] = "no"
    family_asd: Optional[str] = "no"
    answers: List[AnswerItem]

# ==========================================
# 3. LOGIC ENGINES (CORE PROCESSING)
# ==========================================

def process_autism(data: AssessmentRequest):
    """
    Autism Logic:
    - Q1-Q9: No = 1 (Risk), Yes = 0 (Safe)
    - Q10: Yes = 1 (Risk), No = 0 (Safe)
    """
    if "autism" not in models:
        raise HTTPException(status_code=500, detail="Autism model not loaded.")

    ans_map = {item.q_id: item.answer.lower().strip() for item in data.answers}
    input_vector = []
    
    for i in range(1, 11):
        ans = ans_map.get(i, "yes")
        if i == 10:
            val = 1 if ans == "yes" else 0
        else:
            val = 1 if ans == "no" else 0
        input_vector.append(val)
    
    input_vector.append(data.age)
    input_vector.append(1 if data.sex.lower() == 'm' else 0)
    input_vector.append(1 if data.jaundice.lower() == 'yes' else 0)
    input_vector.append(1 if data.family_asd.lower() == 'yes' else 0)
    
    df = pd.DataFrame([input_vector], columns=models["autism_features"])
    
    pred = models["autism"].predict(df)[0]
    prob = models["autism"].predict_proba(df)[0][1]
    
    return {
        "test_type": "Autism",
        "result": "High Risk (ASD)" if pred == 1 else "Low Risk (Normal)",
        "risk_score": round(prob * 100, 2)
    }

def process_adhd(data: AssessmentRequest):
    """
    ADHD Logic (Normalized 0.0 - 1.0):
    - Maps user inputs to Normalized Range [0.0 - 1.0]
    - Calculation: (User Sum) / (Max Possible App Score)
    - Matches the 'adhd_xgb_model_optimized.joblib' trained with normalization.
    """
    if "adhd" not in models:
        raise HTTPException(status_code=500, detail="ADHD model not loaded.")

    raw_map = {}
    for item in data.answers:
        a = item.answer.lower().strip()
        score = 2 if a == "yes" else 1 if a == "sometimes" else 0
        raw_map[item.q_id] = score
    
    def get_s(q_id): return raw_map.get(q_id, 0)

    # --- NORMALIZATION LOGIC (0.0 to 1.0) ---

    # Hyperactivity (Q1-Q3) -> Max 6
    norm_hyper = (get_s(1) + get_s(2) + get_s(3)) / 6.0
    
    # Conduct (Q4-Q5) -> Max 4
    norm_conduct = (get_s(4) + get_s(5)) / 4.0
    
    # Emotional (Q6) -> Max 2
    norm_emotional = get_s(6) / 2.0
    
    # Peer (Q7) -> Max 2
    norm_peer = get_s(7) / 2.0
    
    # Prosocial (Q8) -> Max 2 (High is Good)
    norm_prosocial = get_s(8) / 2.0

    # APQ Parenting (Q9-Q13) -> Each Max 2
    norm_apq_involvement = get_s(9) / 2.0
    norm_apq_positive = get_s(10) / 2.0
    norm_apq_poor_mon = get_s(11) / 2.0
    norm_apq_corporal = get_s(12) / 2.0
    norm_apq_inconsistent = get_s(13) / 2.0
    norm_apq_other = 0.0

    # --- Derived Totals (Normalized) ---
    # Total Difficulties (Sum of raw / Max raw) -> Max 14
    raw_total = (get_s(1)+get_s(2)+get_s(3)) + (get_s(4)+get_s(5)) + get_s(6) + get_s(7)
    norm_total = raw_total / 14.0
    
    # Externalizing -> Max 10
    raw_ext = (get_s(1)+get_s(2)+get_s(3)) + (get_s(4)+get_s(5))
    norm_ext = raw_ext / 10.0
    
    # Internalizing -> Max 4
    raw_int = get_s(6) + get_s(7)
    norm_int = raw_int / 4.0
    
    # Impact (Inferred)
    norm_impact = norm_total

    # --- Construct Input ---
    input_dict = {
        "Hyperactivity_Score": norm_hyper,
        "Conduct_Problems": norm_conduct,
        "Emotional_Problems": norm_emotional,
        "Peer_Problems": norm_peer,
        "Prosocial_Score": norm_prosocial,
        "Total_Difficulties": norm_total,
        "Externalizing_Score": norm_ext,
        "Internalizing_Score": norm_int,
        "Impact_Score": norm_impact,
        "APQ_Involvement": norm_apq_involvement,
        "APQ_Positive_Parenting": norm_apq_positive,
        "APQ_Poor_Monitoring": norm_apq_poor_mon,
        "APQ_Inconsistent_Discipline": norm_apq_inconsistent,
        "APQ_Corporal_Punishment": norm_apq_corporal,
        "APQ_Other_Discipline": norm_apq_other,
        "Age": data.age,
        "Sex": 1 if data.sex.lower() == 'f' else 0
    }
    
    df = pd.DataFrame([input_dict], columns=models["adhd_features"])
    
    pred = models["adhd"].predict(df)[0]
    prob = models["adhd"].predict_proba(df)[0][1]
    
    return {
        "test_type": "ADHD",
        "result": "ADHD Likely" if pred == 1 else "No ADHD Likely",
        "risk_score": round(prob * 100, 2),
        "details": {
            "hyperactivity_level": round(norm_hyper * 100, 1),
            "total_difficulty_level": round(norm_total * 100, 1)
        }
    }

def process_dyslexia(data: AssessmentRequest):
    """
    Dyslexia Logic:
    - Yes(Problem) = 0.0, No(Good) = 1.0
    - Label 0 = High Risk, 1 = Moderate, 2 = Low Risk
    """
    if "dyslexia" not in models:
        raise HTTPException(status_code=500, detail="Dyslexia model not loaded.")

    raw_map = {}
    for item in data.answers:
        a = item.answer.lower().strip()
        # 0.0 is Bad (Risk), 1.0 is Good (Safe)
        score = 0.0 if a == "yes" else 0.5 if a == "sometimes" else 1.0
        raw_map[item.q_id] = score

    def get_avg(q_list):
        vals = [raw_map.get(q, 1.0) for q in q_list]
        return sum(vals) / len(vals) if vals else 0.0

    score_lang = get_avg([1, 2, 3])
    score_mem = get_avg([4, 5])
    score_speed = get_avg([6])
    score_visual = get_avg([7, 8])
    score_audio = get_avg([9, 10])
    score_survey = (score_lang + score_mem + score_speed + score_visual + score_audio) / 5.0

    input_vector = [score_lang, score_mem, score_speed, score_visual, score_audio, score_survey]
    df = pd.DataFrame([input_vector], columns=[
        'Language_vocab', 'Memory', 'Speed', 
        'Visual_discrimination', 'Audio_Discrimination', 'Survey_Score'
    ])
    
    pred = models["dyslexia"].predict(df)[0]
    
    if pred == 0:
        res = "High Risk (Dyslexia)"
    elif pred == 1:
        res = "Moderate Risk"
    else:
        res = "Low Risk (Normal)"
        
    return {
        "test_type": "Dyslexia",
        "result": res,
        "details": {
            "language_score": round(score_lang, 2),
            "memory_score": round(score_mem, 2)
        }
    }

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.post("/predict")
def predict_endpoint(data: AssessmentRequest):
    try:
        tt = data.test_type.lower().strip()
        if tt == "autism":
            return process_autism(data)
        elif tt == "adhd":
            return process_adhd(data)
        elif tt == "dyslexia":
            return process_dyslexia(data)
        else:
            raise HTTPException(status_code=400, detail="Invalid test_type.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logic Error: {str(e)}")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "message": "Psychological Assessment AI is Running",
        "models_loaded": list(models.keys())
    }