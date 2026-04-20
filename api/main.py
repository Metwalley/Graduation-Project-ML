import pandas as pd
import joblib
import numpy as np
import httpx
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ==========================================
# 1. APP CONFIGURATION & MODEL LOADING
# ==========================================
app = FastAPI(
    title="Child Psychological Assessment - Unified AI Engine",
    description="""
## Unified AI API for the Child Psychological Assessment System.

Handles **three core responsibilities** so the Spring Boot backend only needs one base URL:

1. **`POST /predict`** — Initial diagnostic assessment (Autism, ADHD, Dyslexia) using trained ML models.
2. **`POST /monthly-tracker`** — Monthly progress scoring. Calculates trend (improved/stable/declined) vs. previous month.
3. **`POST /chat`** — Educational RAG chatbot, proxied internally to the Ollama service.
4. **`GET /health`** — Service health check.
    """,
    version="2.0.0"
)

# Allow all origins for development/demo — Spring Boot and Flutter can call freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")

AUTISM_MODEL_PATH    = os.path.join(MODELS_DIR, "autism_xgb_model.joblib")
AUTISM_FEATURES_PATH = os.path.join(MODELS_DIR, "autism_features.joblib")
ADHD_MODEL_PATH      = os.path.join(MODELS_DIR, "adhd_xgb_model_optimized.joblib")
DYSLEXIA_MODEL_PATH  = os.path.join(MODELS_DIR, "dyslexia_rf_model.joblib")

# Chatbot service URL — resolved via Docker internal network name or env override
CHATBOT_URL = os.getenv("CHATBOT_SERVICE_URL", "http://chatbot-service:8001")

# Monthly tracker questions JSON — loaded once at startup
TRACKER_JSON_PATH = os.getenv(
    "TRACKER_JSON_PATH",
    os.path.join(BASE_DIR, "..", "seed_data", "monthly_tracker.json")
)

# Global model store
models = {}
tracker_data = {}


def load_models():
    """Loads all ML models into memory at startup."""
    print(f"⏳ Loading AI Models from: {MODELS_DIR} ...")
    try:
        models["autism"]          = joblib.load(AUTISM_MODEL_PATH)
        models["autism_features"] = joblib.load(AUTISM_FEATURES_PATH)
        models["adhd"]            = joblib.load(ADHD_MODEL_PATH)
        models["adhd_features"]   = [
            "Hyperactivity_Score", "Conduct_Problems", "Emotional_Problems",
            "Peer_Problems", "Prosocial_Score", "Total_Difficulties",
            "Externalizing_Score", "Internalizing_Score", "Impact_Score",
            "APQ_Involvement", "APQ_Positive_Parenting", "APQ_Poor_Monitoring",
            "APQ_Inconsistent_Discipline", "APQ_Corporal_Punishment",
            "APQ_Other_Discipline", "Age", "Sex"
        ]
        models["dyslexia"]        = joblib.load(DYSLEXIA_MODEL_PATH)
        print("✅ All ML models loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
    except Exception as e:
        print(f"❌ Critical error loading models: {e}")


def load_tracker_data():
    """Loads the monthly tracker questions JSON at startup."""
    global tracker_data
    try:
        resolved = os.path.abspath(TRACKER_JSON_PATH)
        with open(resolved, "r", encoding="utf-8") as f:
            tracker_data = json.load(f)
        print(f"✅ Monthly tracker data loaded ({len(tracker_data)} disorders).")
    except FileNotFoundError:
        print(f"⚠️  monthly_tracker.json not found at: {resolved}. Tracker endpoint will use default logic.")
    except Exception as e:
        print(f"❌ Error loading tracker data: {e}")


load_models()
load_tracker_data()


# ==========================================
# 2. DATA SCHEMAS (Pydantic)
# ==========================================

# --- Initial Diagnosis ---
class AnswerItem(BaseModel):
    q_id: int
    answer: str  # "yes", "no", "sometimes"

class AssessmentRequest(BaseModel):
    test_type: str          # "autism" | "adhd" | "dyslexia"
    age: float
    sex: str                # "m" | "f"
    jaundice: Optional[str] = "no"
    family_asd: Optional[str] = "no"
    answers: List[AnswerItem]

# --- Monthly Tracker ---
class TrackerAnswerItem(BaseModel):
    q_id: str   # e.g. "m_adhd_1"
    value: int  # 0 = improved, 1 = same, 2 = worse

class MonthlyTrackerRequest(BaseModel):
    disorder: str                    # "adhd" | "autism" | "dyslexia"
    answers: List[TrackerAnswerItem]
    previous_score: Optional[int] = None  # None means this is the first monthly check

class MonthlyTrackerResponse(BaseModel):
    disorder: str
    current_score: int
    max_score: int
    progress_percentage: float
    trend: str           # "improved" | "stable" | "declined" | "baseline"
    trend_label: str     # Arabic label for UI display
    interpretation: str  # Arabic explanation

# --- Chatbot ---
class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    is_safe: bool = True


# ==========================================
# 3. DIAGNOSIS LOGIC ENGINES (unchanged)
# ==========================================

def process_autism(data: AssessmentRequest):
    """
    Autism scoring:
    - Q1-Q9: No = 1 (Risk), Yes = 0 (Safe)
    - Q10:   Yes = 1 (Risk), No  = 0 (Safe)
    """
    if "autism" not in models:
        raise HTTPException(status_code=500, detail="Autism model not loaded.")

    ans_map = {item.q_id: item.answer.lower().strip() for item in data.answers}
    input_vector = []

    for i in range(1, 11):
        ans = ans_map.get(i, "yes")
        val = (1 if ans == "yes" else 0) if i == 10 else (1 if ans == "no" else 0)
        input_vector.append(val)

    input_vector.append(data.age)
    input_vector.append(1 if data.sex.lower() == "m" else 0)
    input_vector.append(1 if data.jaundice.lower() == "yes" else 0)
    input_vector.append(1 if data.family_asd.lower() == "yes" else 0)

    df   = pd.DataFrame([input_vector], columns=models["autism_features"])
    pred = models["autism"].predict(df)[0]
    prob = models["autism"].predict_proba(df)[0][1]

    return {
        "test_type":  "Autism",
        "result":     "High Risk (ASD)" if pred == 1 else "Low Risk (Normal)",
        "risk_score": round(prob * 100, 2),
    }


def process_adhd(data: AssessmentRequest):
    """
    ADHD scoring with Proportional Normalization (0.0 → 1.0).
    This is the critical architectural decision — do NOT change without review.
    """
    if "adhd" not in models:
        raise HTTPException(status_code=500, detail="ADHD model not loaded.")

    raw_map = {}
    for item in data.answers:
        a = item.answer.lower().strip()
        raw_map[item.q_id] = 2 if a == "yes" else 1 if a == "sometimes" else 0

    def s(q): return raw_map.get(q, 0)

    norm_hyper     = (s(1) + s(2) + s(3)) / 6.0
    norm_conduct   = (s(4) + s(5))        / 4.0
    norm_emotional = s(6)                  / 2.0
    norm_peer      = s(7)                  / 2.0
    norm_prosocial = s(8)                  / 2.0

    norm_apq_involvement  = s(9)  / 2.0
    norm_apq_positive     = s(10) / 2.0
    norm_apq_poor_mon     = s(11) / 2.0
    norm_apq_corporal     = s(12) / 2.0
    norm_apq_inconsistent = s(13) / 2.0
    norm_apq_other        = 0.0

    raw_total  = s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)
    norm_total = raw_total / 14.0
    norm_ext   = (s(1)+s(2)+s(3)+s(4)+s(5)) / 10.0
    norm_int   = (s(6)+s(7)) / 4.0
    norm_impact = norm_total

    input_dict = {
        "Hyperactivity_Score":       norm_hyper,
        "Conduct_Problems":          norm_conduct,
        "Emotional_Problems":        norm_emotional,
        "Peer_Problems":             norm_peer,
        "Prosocial_Score":           norm_prosocial,
        "Total_Difficulties":        norm_total,
        "Externalizing_Score":       norm_ext,
        "Internalizing_Score":       norm_int,
        "Impact_Score":              norm_impact,
        "APQ_Involvement":           norm_apq_involvement,
        "APQ_Positive_Parenting":    norm_apq_positive,
        "APQ_Poor_Monitoring":       norm_apq_poor_mon,
        "APQ_Inconsistent_Discipline": norm_apq_inconsistent,
        "APQ_Corporal_Punishment":   norm_apq_corporal,
        "APQ_Other_Discipline":      norm_apq_other,
        "Age":                       data.age,
        "Sex":                       1 if data.sex.lower() == "f" else 0,
    }

    df   = pd.DataFrame([input_dict], columns=models["adhd_features"])
    pred = models["adhd"].predict(df)[0]
    prob = models["adhd"].predict_proba(df)[0][1]

    return {
        "test_type":  "ADHD",
        "result":     "ADHD Likely" if pred == 1 else "No ADHD Likely",
        "risk_score": round(prob * 100, 2),
        "details": {
            "hyperactivity_level":    round(norm_hyper * 100, 1),
            "total_difficulty_level": round(norm_total * 100, 1),
        },
    }


def process_dyslexia(data: AssessmentRequest):
    """
    Dyslexia scoring (Reverse Logic):
    Yes (struggle) = 0.0 (risk), No (fine) = 1.0 (safe).
    """
    if "dyslexia" not in models:
        raise HTTPException(status_code=500, detail="Dyslexia model not loaded.")

    raw_map = {}
    for item in data.answers:
        a = item.answer.lower().strip()
        raw_map[item.q_id] = 0.0 if a == "yes" else 0.5 if a == "sometimes" else 1.0

    def avg(q_list):
        vals = [raw_map.get(q, 1.0) for q in q_list]
        return sum(vals) / len(vals)

    score_lang   = avg([1, 2, 3])
    score_mem    = avg([4, 5])
    score_speed  = avg([6])
    score_visual = avg([7, 8])
    score_audio  = avg([9, 10])
    score_survey = (score_lang + score_mem + score_speed + score_visual + score_audio) / 5.0

    df   = pd.DataFrame(
        [[score_lang, score_mem, score_speed, score_visual, score_audio, score_survey]],
        columns=["Language_vocab", "Memory", "Speed", "Visual_discrimination", "Audio_Discrimination", "Survey_Score"],
    )
    pred = models["dyslexia"].predict(df)[0]

    result_map = {0: "High Risk (Dyslexia)", 1: "Moderate Risk", 2: "Low Risk (Normal)"}

    return {
        "test_type": "Dyslexia",
        "result":    result_map.get(pred, "Unknown"),
        "details": {
            "language_score": round(score_lang, 2),
            "memory_score":   round(score_mem, 2),
        },
    }


# ==========================================
# 4. MONTHLY TRACKER ENGINE
# ==========================================

# Maps disorder key to its title-cased key in the JSON
DISORDER_KEY_MAP = {
    "adhd":     "ADHD",
    "autism":   "Autism",
    "dyslexia": "Dyslexia",
}

# Max score = 5 questions × max value 2
MAX_SCORE_PER_DISORDER = 10


def calculate_trend(current: int, previous: Optional[int]) -> dict:
    """
    Determines the trend label and interpretation based on score delta.
    
    Scoring direction: LOWER is BETTER (0 = all improved, 10 = all declined).
    
    Thresholds (from monthly_tracker.json):
        improved : previous - current >= 2
        stable   : |previous - current| <= 1
        declined : current - previous >= 2
        baseline : no previous score (first monthly check)
    """
    if previous is None:
        return {
            "trend":          "baseline",
            "trend_label":    "تقييم أساسي 📊",
            "interpretation": "هذا هو أول تقييم شهري. ستُستخدم هذه النتيجة كخط أساسي لمقارنة التقدم في الأشهر القادمة.",
        }

    delta = previous - current  # positive delta = score went DOWN = improvement

    if delta >= 2:
        return {
            "trend":          "improved",
            "trend_label":    "تحسن ملحوظ 📈",
            "interpretation": f"انخفض مستوى القلق بمقدار {delta} نقاط مقارنة بالشهر الماضي. استمر على نفس النهج!",
        }
    elif delta <= -2:
        return {
            "trend":          "declined",
            "trend_label":    "تراجع يستدعي الانتباه 📉",
            "interpretation": f"ارتفع مستوى القلق بمقدار {abs(delta)} نقاط مقارنة بالشهر الماضي. قد يكون مفيداً مراجعة الأخصائي.",
        }
    else:
        return {
            "trend":          "stable",
            "trend_label":    "مستقر 📊",
            "interpretation": "لم يتغير مستوى القلق بشكل ملحوظ هذا الشهر. تابع الأنشطة والتمارين الموصى بها.",
        }


def process_monthly_tracker(req: MonthlyTrackerRequest) -> MonthlyTrackerResponse:
    disorder_key = req.disorder.lower().strip()
    if disorder_key not in DISORDER_KEY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid disorder '{req.disorder}'. Must be one of: adhd, autism, dyslexia."
        )

    # Validate answer values
    for ans in req.answers:
        if ans.value not in (0, 1, 2):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value '{ans.value}' for question '{ans.q_id}'. Must be 0, 1, or 2."
            )

    # Validate answer count — must be exactly 5 (one per question)
    # Without this, a partial submission would produce a misleading score out of 10
    expected_count = 5
    if len(req.answers) != expected_count:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {expected_count} answers for '{req.disorder}', "
                f"got {len(req.answers)}. "
                f"All 5 questions must be answered."
            )
        )

    # Calculate current score (simple sum — exactly as defined in the JSON)
    current_score = sum(ans.value for ans in req.answers)
    current_score = min(current_score, MAX_SCORE_PER_DISORDER)  # safety cap

    # Calculate progress percentage (inverted: lower score = higher progress)
    # 0 score = 100% progress, 10 score = 0% progress
    progress_percentage = round(((MAX_SCORE_PER_DISORDER - current_score) / MAX_SCORE_PER_DISORDER) * 100, 1)

    trend_info = calculate_trend(current_score, req.previous_score)

    return MonthlyTrackerResponse(
        disorder=DISORDER_KEY_MAP[disorder_key],
        current_score=current_score,
        max_score=MAX_SCORE_PER_DISORDER,
        progress_percentage=progress_percentage,
        trend=trend_info["trend"],
        trend_label=trend_info["trend_label"],
        interpretation=trend_info["interpretation"],
    )


# ==========================================
# 5. API ENDPOINTS
# ==========================================

@app.get("/health", tags=["System"])
def health_check():
    """Service health check. Returns which models are currently loaded."""
    return {
        "status":        "online",
        "version":       "2.0.0",
        "models_loaded": [k for k in models if not k.endswith("_features")],
        "tracker_loaded": bool(tracker_data),
        "chatbot_url":   CHATBOT_URL,
    }


@app.post("/predict", tags=["1. Initial Diagnosis"])
def predict_endpoint(data: AssessmentRequest):
    """
    **Initial Diagnostic Assessment**

    Runs the child's questionnaire answers through the trained ML model
    for the specified disorder and returns a prediction with confidence score.

    - `test_type`: `"autism"` | `"adhd"` | `"dyslexia"`
    - `answers`: list of `{ q_id, answer }` where answer is `"yes"` | `"no"` | `"sometimes"`
    """
    tt = data.test_type.lower().strip()
    if tt == "autism":
        return process_autism(data)
    elif tt == "adhd":
        return process_adhd(data)
    elif tt == "dyslexia":
        return process_dyslexia(data)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid test_type '{data.test_type}'. Use: autism, adhd, dyslexia.")


@app.post("/monthly-tracker", response_model=MonthlyTrackerResponse, tags=["2. Monthly Progress Tracker"])
def monthly_tracker_endpoint(req: MonthlyTrackerRequest):
    """
    **Monthly Progress Tracking**

    Accepts the parent's monthly check-in answers and returns a scored result
    with a trend indicator compared to the previous month.

    **Scoring:** Lower score = more improvement (0 = best, 10 = worst).

    **Answer values:**
    - `0` = "أفضل بكثير" (Much better)
    - `1` = "نفس الحال" (Same as before)
    - `2` = "أسوأ" (Worse)

    **Trend logic:**
    - `improved`  → previous_score − current_score ≥ 2
    - `stable`    → |previous_score − current_score| ≤ 1
    - `declined`  → current_score − previous_score ≥ 2
    - `baseline`  → no previous_score (first monthly check)
    """
    return process_monthly_tracker(req)


@app.get("/monthly-tracker/questions/{disorder}", tags=["2. Monthly Progress Tracker"])
def get_tracker_questions(disorder: str):
    """
    **Get Monthly Tracker Questions**

    Returns the full list of monthly check-in questions for a given disorder.
    Flutter uses this to render the questionnaire UI dynamically.

    - `disorder`: `adhd` | `autism` | `dyslexia`
    """
    disorder_key = disorder.lower().strip()
    json_key = DISORDER_KEY_MAP.get(disorder_key)

    if not json_key:
        raise HTTPException(status_code=400, detail=f"Invalid disorder '{disorder}'.")

    if not tracker_data:
        raise HTTPException(status_code=503, detail="Tracker data not loaded on server.")

    if json_key not in tracker_data:
        raise HTTPException(status_code=404, detail=f"No tracker data found for '{disorder}'.")

    return tracker_data[json_key]


@app.post("/chat", response_model=ChatResponse, tags=["3. Educational Chatbot"])
async def chat_endpoint(req: ChatRequest):
    """
    **Educational Chatbot (RAG-powered)**

    Proxies the request to the internal RAG chatbot service (Ollama + Llama 3.2).
    Spring Boot does not need to know the chatbot's internal address.

    - Refuses medical advice and prescriptions (built-in safety filter).
    - Responds in Arabic with educational information only.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{CHATBOT_URL}/chat",
                # Chatbot schema uses 'question', not 'text'
                json={"question": req.text},
            )
            response.raise_for_status()
            data = response.json()
            # Chatbot returns 'answer'; blocked=True means unsafe query
            is_safe = not data.get("blocked", False)
            return ChatResponse(
                response=data.get("answer", ""),
                is_safe=is_safe,
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Chatbot service is currently unavailable. Ensure the chatbot container is running."
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Chatbot service timed out. The model may still be loading."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot proxy error: {str(e)}")