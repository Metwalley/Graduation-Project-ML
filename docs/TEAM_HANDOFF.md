# الـ AI Engine — دليل التشغيل والتكامل
### كتبه: عبدالرحمن | آخر تحديث: مايو 2026

---

> اقرا الملف ده كامل قبل ما تمس أي حاجة. فيه كل حاجة محتاجها.

---

## الصورة الكاملة — إيه اللي عملته؟

عملت **API واحد موحد** على بورت `8000` بيعمل 4 حاجات:

| الوظيفة | الـ Endpoint |
|---|---|
| التشخيص الأولي للطفل | `POST /predict` |
| التقييم الشهري (حساب التقدم) | `POST /monthly-tracker` |
| جيب أسئلة الشهر | `GET /monthly-tracker/questions/{disorder}` |
| الشات بوت التعليمي | `POST /chat` |
| فحص حالة الـ API | `GET /health` |

**إزاي بيشتغل مع بعض:**

```
أهل الطفل (Flutter App)
         ↓
Spring Boot Backend
         ↓
AI Engine — port 8000 (ده اللي عملته أنا)
```

Flutter مش بتكلم الـ AI Engine مباشرة — بتكلم Spring Boot، وSpring Boot بيكلم الـ AI Engine.

---

## إيه اللي محتاجينه

**متطلب وحيد:** Docker Desktop مثبت وشغال على الجهاز.
حمّله من: https://www.docker.com/products/docker-desktop/

---

## الملفات اللي هتاخدوها من الريبو

```
Graduation-Project-ML/
│
├── docker-compose.yml      ← ده اللي بيشغّل كل حاجة (أهم ملف)
│
├── .env                    ← هتعملوه انتوا (مش موجود في الريبو — فيه secrets)
│
├── seed_data/
│   ├── articles.json       ← مقالات للعرض في الـ App
│   ├── exercises.json      ← تمارين، ازرعوها في MySQL
│   └── monthly_tracker.json← أسئلة التقييم الشهري
│
└── docs/
    └── TEAM_HANDOFF.md     ← الملف اللي بتقراه دلوقتي
```

مش محتاجين: `api/` ولا `chatbot_service/` ولا `models/` ولا `notebooks/` — دي كلها جوه الـ Docker Image بالفعل.

---

## تشغيل الـ API — خطوتين بس

### الخطوة 1 — اعمل ملف `.env` في نفس فولدر `docker-compose.yml`

الملف اسمه `.env` (بالنقطة في الأول) وفيه سطر واحد بس:

```
GROQ_API_KEY=اطلب الـ key من عبدالرحمن
```

**شكل الملفات بعد ما تعمل الـ .env:**
```
Graduation-Project-ML/
├── docker-compose.yml   ✅ موجود من الريبو
├── .env                 ✅ انت عملته دلوقتي
└── seed_data/           ✅ موجود من الريبو
```

### الخطوة 2 — شغّل الـ API

```bash
docker-compose up
```

أول مرة بس هياخد دقيقتين يحمّل الـ Image (~400MB).
من بعدين هيجي في ثواني.

### تأكد إنه شغال:

افتح المتصفح:
```
http://localhost:8000/health
```

لازم يطلع:
```json
{
  "status": "online",
  "models_loaded": ["autism", "adhd", "dyslexia"],
  "chatbot_ready": true
}
```

لو طلع كده — كل حاجة شغالة. خلاص.

---

## Swagger UI — اختبار الـ Endpoints من المتصفح

```
http://localhost:8000/docs
```

من هنا تقدروا تجربوا كل endpoint مباشرة من المتصفح بدون Postman.

---

## تكامل الـ API مع Spring Boot

### الإعداد

في `application.properties` أو `application.yml`:
```properties
ai.engine.url=http://localhost:8000
```

### مثال — التشخيص الأولي:

```java
@Service
public class AiEngineService {

    @Value("${ai.engine.url}")
    private String aiEngineUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    public DiagnosisResult predict(PredictRequest request) {
        String url = aiEngineUrl + "/predict";
        ResponseEntity<DiagnosisResult> response =
            restTemplate.postForEntity(url, request, DiagnosisResult.class);
        return response.getBody();
    }

    public ChatResult chat(String userMessage) {
        String url = aiEngineUrl + "/chat";
        Map<String, String> body = Map.of("text", userMessage);
        ResponseEntity<ChatResult> response =
            restTemplate.postForEntity(url, body, ChatResult.class);
        return response.getBody();
    }

    public MonthlyResult monthlyTracker(MonthlyRequest request) {
        String url = aiEngineUrl + "/monthly-tracker";
        ResponseEntity<MonthlyResult> response =
            restTemplate.postForEntity(url, request, MonthlyResult.class);
        return response.getBody();
    }
}
```

---

## كيفية التكامل مع كل Endpoint

---

### 1. التشخيص الأولي — `POST /predict`

**Request:**
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
GET http://localhost:8000/monthly-tracker/questions/adhd
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
        { "text": "أسوأ",       "value": 2 }
      ]
    }
  ]
}
```

Flutter يجيب الأسئلة دي ويعرضها. بعد ما الأب يجاوب، يبعتها لـ `POST /monthly-tracker`.

---

### 3. التقييم الشهري — `POST /monthly-tracker`

**المسار الكامل:**
```
الأب يجاوب الأسئلة في الـ App (Flutter)
    ↓
Flutter يبعت للـ Spring Boot
    ↓
Spring Boot يجيب آخر score من DB للطفل ده
    ↓
Spring Boot يبعت للـ AI Engine
    ↓
AI Engine يحسب التقدم ويرجع النتيجة
    ↓
Spring Boot يحفظ الـ score الجديد في DB
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

> **`previous_score`:** آخر `current_score` عندكم في الـ DB للطفل ده. لو أول شهر → ابعت `null`.

> **مهم:** لازم تبعتوا 5 إجابات بالظبط، مش أقل ومش أكتر — هيرجع 422 لو غلط.

**قيم `value`:** `0` = أفضل | `1` = نفس الحال | `2` = أسوأ

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

**قيم `trend`:** `improved` | `stable` | `declined` | `baseline`

---

### 4. الشات بوت — `POST /chat`

**Request:**
```json
{
  "text": "ابني مش بيركز في المذاكرة، اعمل معاه إيه؟"
}
```

**Response — سؤال عادي:**
```json
{
  "response": "يمكنك اتباع الخطوات التالية للمساعدة في تحسين تركيز طفلك...",
  "is_safe": true
}
```

**Response — لو السؤال فيه طلب دواء:**
```json
{
  "response": "أنا مساعد تعليمي فقط ولا أستطيع تقديم وصفات طبية. يرجى التواصل مع طبيب متخصص.",
  "is_safe": false
}
```

الشات بوت بياخد **1-3 ثواني** للرد — بيشتغل على Groq API (محتاج انترنت).

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

-- المقالات (seed من seed_data/articles.json)
CREATE TABLE articles (
  id        INT PRIMARY KEY,
  category  VARCHAR(20),
  title     TEXT,
  summary   TEXT,
  content   LONGTEXT,
  read_time VARCHAR(20)
);

-- التمارين (seed من seed_data/exercises.json)
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

### Endpoints مطلوبة من Spring Boot:

```
GET /api/articles?disorder=ADHD       → قائمة مقالات
GET /api/articles/{id}                → مقالة كاملة
GET /api/exercises?disorder=ADHD      → تمارين
GET /api/children/{id}/monthly-scores → تاريخ التقدم للـ Line Chart
```

---

## الـ Line Chart — إزاي Flutter يرسمه

Spring Boot يرجع:
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

## عرض المشروع على الموبايل يوم المناقشة

مش محتاجين Cloud. الحل:

```
التليفون + اللابتوب على نفس الـ WiFi
شغّل docker-compose up على اللابتوب
Spring Boot يتصل بـ IP اللابتوب بدل localhost
```

**عرف الـ IP:**
```powershell
ipconfig | findstr "IPv4"
```

في `application.properties`:
```properties
ai.engine.url=http://192.168.1.x:8000
```

---

## يوم المناقشة ⚠️

> - **قبل يوم:** شغّل `docker-compose up` مرة وتأكد إن `/health` بيرجع صح.
> - **يوم المناقشة:** `docker-compose up` وهيجي في ثواني — الـ Image محملة.
> - **⚠️ الشات بوت محتاج انترنت** — تأكد في الكلية إن في WiFi أو هوت سبوت.
> - **التشخيص والـ Monthly Tracker شغالين offline** — مش محتاجين انترنت.

---

## ملخص المهام

| المهمة | المسؤول | الحالة |
|---|---|---|
| موديلات التشخيص (3 disorders) | Abdo | ✅ |
| الشات بوت (Groq API) | Abdo | ✅ |
| التقييم الشهري (API + أسئلة) | Abdo | ✅ |
| Docker Image على Docker Hub | Abdo | ✅ |
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
