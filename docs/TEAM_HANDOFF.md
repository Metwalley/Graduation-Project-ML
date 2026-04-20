# الـ AI Engine — دليل التشغيل والتكامل
### كتبه: عبدالرحمن | آخر تحديث: أبريل 2026

---

> اقرا الملف ده كامل قبل ما تمس أي حاجة. فيه كل حاجة محتاجها.

---

## إيه اللي عملته؟

عملت **API واحد موحد** على بورت `8000` بدل ما كل service ليها بورت لوحدها. يعني انتوا (الباك إند) بتكلموا عنوان واحد بس وخلاص.

| الوظيفة | الـ Endpoint |
|---|---|
| التشخيص الأولي | `POST /predict` |
| التقييم الشهري | `POST /monthly-tracker` |
| أسئلة الشهر | `GET /monthly-tracker/questions/{disorder}` |
| الشات بوت | `POST /chat` |
| فحص الحالة | `GET /health` |

كل ده على: `http://ai-engine:8000`

---

## الملفات اللي هتاخدوها

```
Graduation-Project-ML/
│
├── api/                    ← ده الأهم، فيه كل الـ endpoints
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── ml_models/          ← ملفات الموديلات، ملمسوش
│
├── chatbot_service/        ← الشات بوت، Docker بيشغّله تلقائياً
│
├── seed_data/
│   ├── articles.json       ← مقالات للشات بوت + للعرض في الـ App
│   ├── exercises.json      ← ازرعوها في MySQL
│   └── monthly_tracker.json← أسئلة التقييم الشهري
│
├── docker-compose.yml      ← ده اللي بيشغّل كل حاجة
└── docs/TEAM_HANDOFF.md    ← الملف اللي بتقراه دلوقتي
```

مش محتاجين: `models/` ولا `notebooks/` ولا `data/`

---

## تشغيل بـ Docker — ده الأسهل والمضمون

### شرط وحيد: Docker Desktop مثبت وشغال

```bash
cd Graduation-Project-ML
docker-compose up --build
```

### ⚠️ أول تشغيل بس
بيحمّل Llama (~2GB). هياخد وقت. من بعدين هيجي بسرعة.

### تأكد إنه شغال:
```
http://localhost:8000/health
```
لازم يطلع: `"models_loaded": ["autism", "adhd", "dyslexia"]`

---

## تشغيل يدوي — لو Docker مش شغال معاك

محتاج **3 terminals منفصلة بالترتيب ده:**

**Terminal 1 — Ollama:**
```powershell
ollama serve
```
تأكد إن الموديل موجود:
```powershell
ollama list
# لو مش موجود:
ollama pull llama3.2:3b
```

**Terminal 2 — الشات بوت (لازم يشتغل على بورت 8001):**
```powershell
cd chatbot_service
$env:TF_ENABLE_ONEDNN_OPTS="0"
uvicorn main:app --port 8001
```
استنى لحد ما يطلع: `RAG Engine ready!`

**Terminal 3 — الـ AI Engine:**
```powershell
cd api
$env:CHATBOT_SERVICE_URL="http://localhost:8001"
uvicorn main:app --port 8000
```
استنى لحد ما يطلع: `✅ All ML models loaded successfully.`

### ⚠️ مشاكل شائعة وحلها

| المشكلة | السبب | الحل |
|---|---|---|
| `/chat` بيرجع `503` | نسيت تعمل `CHATBOT_SERVICE_URL` | ابعته قبل تشغيل الـ API (Terminal 3) |
| `paging file too small` | RAM ممتلئة | اقفل terminals قديمة وابدأ من أول |
| `Port already in use` | في service تاني على نفس البورت | `netstat -ano` تلاقي الـ PID |
| الشات بوت بيأخد وقت | الموديل بيشتغل على CPU | طبيعي، 5-15 ثانية |

---

## كيفية التكامل مع كل Endpoint

---

### 1. التشخيص الأولي — `POST /predict`

```json
{
  "test_type": "adhd",
  "age": 8,
  "sex": "m",
  "answers": [
    { "q_id": 1, "answer": "yes" },
    { "q_id": 2, "answer": "sometimes" },
    { "q_id": 3, "answer": "no" }
  ]
}
```

**قيم `test_type`:** `"autism"` | `"adhd"` | `"dyslexia"`

**قيم `answer`:** `"yes"` | `"no"` | `"sometimes"`

**Response:**
```json
{
  "test_type": "ADHD",
  "result": "ADHD Likely",
  "risk_score": 85.2,
  "details": {
    "hyperactivity_level": 90.5,
    "total_difficulty_level": 78.3
  }
}
```

احفظ `result` + `risk_score` في الـ DB مع بيانات الطفل.

---

### 2. أسئلة التقييم الشهري — `GET /monthly-tracker/questions/{disorder}`

```
GET http://ai-engine:8000/monthly-tracker/questions/adhd
```

**قيم `disorder`:** `adhd` | `autism` | `dyslexia`

**Response:**
```json
{
  "title": "تقييم التقدم الشهري - ADHD",
  "questions": [
    {
      "id": "m_adhd_1",
      "text": "هل لاحظت تحسناً في قدرة طفلك على التركيز؟",
      "type": "scale",
      "options": [
        { "text": "أفضل بكثير", "value": 0 },
        { "text": "نفس الحال",  "value": 1 },
        { "text": "أسوأ",        "value": 2 }
      ]
    }
  ]
}
```

Flutter يجيب الأسئلة دي ويعرضها. بعد ما الأب يجاوب، بعت لـ `POST /monthly-tracker`.

---

### 3. التقييم الشهري — `POST /monthly-tracker`

**المسار الكامل:**
```
أب يجاوب الأسئلة في الـ App
    ↓
Flutter يبعت للـ Backend
    ↓
Backend يجيب آخر score من DB للطفل ده
    ↓
Backend يبعت للـ AI Engine
    ↓
AI Engine يحسب التقدم ويرجع النتيجة
    ↓
Backend يحفظ الـ score الجديد في DB
    ↓
Flutter يعرض النتيجة
```

**Request:**
```json
{
  "disorder": "adhd",
  "answers": [
    { "q_id": "m_adhd_1", "value": 0 },
    { "q_id": "m_adhd_2", "value": 0 },
    { "q_id": "m_adhd_3", "value": 1 },
    { "q_id": "m_adhd_4", "value": 0 },
    { "q_id": "m_adhd_5", "value": 1 }
  ],
  "previous_score": 8
}
```

> **`previous_score`:** آخر `current_score` عندكم في الـ DB. لو أول شهر → ابعت `null`.

> **مهم:** لازم تبعتوا 5 إجابات بالظبط، أقل من كده هيرجع 422.

**قيم `value`:** `0` = أفضل | `1` = نفس | `2` = أسوأ

**Response:**
```json
{
  "disorder": "ADHD",
  "current_score": 2,
  "max_score": 10,
  "progress_percentage": 80.0,
  "trend": "improved",
  "trend_label": "تحسن ملحوظ 📈",
  "interpretation": "انخفض مستوى القلق بمقدار 6 نقاط مقارنة بالشهر الماضي."
}
```

احفظ `current_score` + `disorder` + `assessed_at` في الـ DB.

---

### 4. الشات بوت — `POST /chat`

```json
{
  "text": "ابني مش بيركز في المذاكرة، اعمل معاه إيه؟"
}
```

**Response:**
```json
{
  "response": "يمكنك اتباع الخطوات التالية...",
  "is_safe": true
}
```

لو السؤال فيه طلب دواء أو تشخيص → `is_safe: false` + رسالة رفض مهذبة.

الشات بوت بياخد 5-15 ثانية للرد — طبيعي.

---

## مهام الباك إند اللي محتاجة تتعمل

### DB:

```sql
-- التشخيص
ALTER TABLE children ADD COLUMN disorder VARCHAR(20);
ALTER TABLE children ADD COLUMN risk_score DECIMAL(5,2);

-- التقدم الشهري
CREATE TABLE monthly_scores (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  child_id    INT NOT NULL,
  disorder    VARCHAR(20) NOT NULL,
  score       INT NOT NULL,
  assessed_at DATE NOT NULL
);

-- المقالات (seed من articles.json)
CREATE TABLE articles (
  id        INT PRIMARY KEY,
  category  VARCHAR(20),
  title     TEXT,
  summary   TEXT,
  content   LONGTEXT,
  read_time VARCHAR(20)
);

-- التمارين (seed من exercises.json)
CREATE TABLE exercises (
  id               INT PRIMARY KEY,
  category         VARCHAR(20),
  title            TEXT,
  type             VARCHAR(50),
  difficulty       VARCHAR(20),
  duration_minutes INT,
  description      TEXT,
  goal             TEXT
);
```

### Endpoints مطلوبة:

```
GET /api/articles?disorder=ADHD       → قائمة مقالات
GET /api/articles/{id}                → مقالة كاملة
GET /api/exercises?disorder=ADHD      → تمارين
GET /api/children/{id}/monthly-scores → تاريخ التقدم للـ Line Chart
```

---

## الـ Line Chart — إزاي Flutter يرسمه

Backend يرجع:
```json
[
  { "month": "2025-01", "progress_percentage": 20.0 },
  { "month": "2025-02", "progress_percentage": 40.0 },
  { "month": "2025-03", "progress_percentage": 80.0 }
]
```

Flutter بيرسم `progress_percentage` على المحور Y، `month` على المحور X.
**فوق = تحسن ✅**

---

## عرض المشروع على الموبايل

مش محتاجين Cloud للـ showcase. الحل الأبسط:

```
التليفون + اللابتوب على نفس الـ WiFi
شغّل docker-compose up على اللابتوب
Flutter يتصل بـ IP اللابتوب بدل localhost
```

**عرف الـ IP:**
```powershell
ipconfig | findstr "IPv4"
```

**في الـ Spring Boot:** غيّر `AI_ENGINE_URL` من `localhost` للـ IP.
**في Flutter:** غيّر الـ base URL للـ IP.

**الـ architecture بتاعي متوافق 100% مع أي Linux server** — لو قررتوا تعلوا cloud في أي وقت، `docker-compose up` وخلاص.

---

## Swagger UI

```
http://localhost:8000/docs
```

من هنا تقدروا تجربوا كل endpoint من المتصفح بدون Postman.

---

## يوم المناقشة ⚠️

> - **قبل يوم كامل:** شغّل `docker-compose up` وخلّيه يحمّل الموديل.
> - **اتأكد** إن `/health` بيرجع كل الـ models.
> - **يوم المناقشة:** `docker-compose up` عادي — الموديل في الـ cache.
> - **الشات بوت شغال offline بالكامل** — مش محتاج انترنت يوم العرض.

---

## ملخص المهام

| المهمة | المسؤول | الحالة |
|---|---|---|
| موديلات التشخيص (3 disorders) | Abdo | ✅ |
| الشات بوت RAG + Ollama | Abdo | ✅ |
| التقييم الشهري (API + أسئلة) | Abdo | ✅ |
| Docker Compose موحد | Abdo | ✅ |
| حفظ نتايج التشخيص في DB | Backend | 🔄 |
| Seed المقالات والتمارين في MySQL | Backend | 🔄 |
| `/articles` و `/exercises` endpoints | Backend | 🔄 |
| ربط `/predict`, `/monthly-tracker`, `/chat` | Backend | 🔄 |
| حفظ monthly scores + `/monthly-scores` endpoint | Backend | 🔄 |
| عرض المقالات والتمارين | Flutter | 🔄 |
| Line Chart التقدم الشهري | Flutter | 🔄 |
| عرض أسئلة الشهر | Flutter | 🔄 |

---

> **أي سؤال أو مشكلة؟ كلموا عبدالرحمن.**
