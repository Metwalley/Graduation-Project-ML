# نظام التقييم النفسي للأطفال — AI Engine

**Graduation Project | Faculty of Computer Science**

نظام ذكاء اصطناعي متكامل للكشف المبكر عن الاضطرابات النمائية عند الأطفال ومتابعة تقدمهم.

---

## 🧠 ما الذي يفعله هذا النظام؟

يوفر هذا النظام **أربع قدرات أساسية** عبر API موحد على بورت واحد:

| الميزة | الوصف | الدقة |
|---|---|---|
| **تشخيص التوحد** | نموذج XGBoost يحلل نتائج استبيان Q-Chat-10 | ~99% |
| **تشخيص ADHD** | نموذج XGBoost مع Proportional Normalization | ~76% |
| **تشخيص عسر القراءة** | نموذج Random Forest | — |
| **شات بوت تعليمي** | Groq API — Llama 3.3 70B — استجابة في 1-3 ثوانٍ | — |
| **متتبع التقدم الشهري** | تسجيل وقياس تحسن الطفل شهرياً | — |

---

## 🚀 تشغيل النظام (دقيقتين)

### المتطلبات
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) مثبت وشغال

### الخطوة 1 — اعمل ملف `.env` في نفس فولدر `docker-compose.yml`

```
GROQ_API_KEY=اطلبها من عبدالرحمن
```

### الخطوة 2 — شغّل

```bash
docker-compose up
```

> أول مرة بس هياخد دقيقة يحمّل الـ Image (~400MB). من بعدين هيجي في ثواني.

### التحقق من التشغيل
```
http://localhost:8000/health   → يرجع: models_loaded + chatbot_ready: true
http://localhost:8000/docs     → Swagger UI تقدر تجرب منه كل endpoint
```

---

## 📡 الـ API Endpoints

**Base URL:** `http://localhost:8000`

| Method | Endpoint | الوظيفة |
|---|---|---|
| `POST` | `/predict` | التشخيص الأولي (autism / adhd / dyslexia) |
| `POST` | `/monthly-tracker` | حساب التقدم الشهري |
| `GET` | `/monthly-tracker/questions/{disorder}` | جلب أسئلة التقييم |
| `POST` | `/chat` | الشات بوت التعليمي (عربي) |
| `GET` | `/health` | فحص حالة الخدمة |

**التوثيق الكامل مع أمثلة وكود Spring Boot:** [`docs/TEAM_HANDOFF.md`](docs/TEAM_HANDOFF.md)

---

## 🏗️ المعمارية

```
Flutter App
    ↓
Spring Boot Backend
    ↓
AI Engine — port 8000   (Docker Image: metwalley/ai-engine:latest)
    ├── POST /predict        → ML models (.joblib — locally in image)
    ├── POST /monthly-tracker → scoring engine
    └── POST /chat           → Groq API (Llama 3.3 70B)
```

Flutter لا تتصل بالـ AI Engine مباشرة — تتصل بـ Spring Boot، وSpring Boot يتصل بالـ AI Engine.

---

## 🗂️ هيكل المشروع

```
Graduation-Project-ML/
│
├── api/                        ← Unified FastAPI (المشغّل الرئيسي)
│   ├── main.py                 ← كل الـ endpoints (predict + tracker + chat)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── ml_models/              ← ملفات الموديلات (.joblib)
│
├── seed_data/                  ← البيانات المشتركة
│   ├── articles.json           ← مقالات للعرض في الـ App
│   ├── exercises.json          ← تمارين وأنشطة — ازرعوها في MySQL
│   └── monthly_tracker.json    ← أسئلة التقييم الشهري
│
├── models/                     ← Training workspace (للـ ML engineer فقط)
│   ├── adhd/
│   ├── autism/
│   └── dyslexia/
│
├── notebooks/                  ← Jupyter notebooks للتدريب والتحليل
│
├── docs/
│   ├── TEAM_HANDOFF.md         ← ← ← ابدأ بده (دليل التكامل الكامل)
│   └── ...
│
├── docker-compose.yml          ← يشغّل كل حاجة بأمر واحد
└── .env                        ← انت بتعمله (فيه GROQ_API_KEY)
```

---

## 🔗 التكامل مع باقي الفريق

**للـ Backend (Spring Boot):** كل الـ endpoints موثقة في [`docs/TEAM_HANDOFF.md`](docs/TEAM_HANDOFF.md) مع أمثلة كود Java جاهزة.

**للـ Flutter:** التطبيق يتصل بـ Spring Boot فقط — Spring Boot هو اللي يتصل بالـ AI Engine.

**للـ ML Engineer:** الموديلات والـ trainers في `models/`، الـ API جاهز في `api/`.

---

## ⚠️ يوم المناقشة

- التشخيص والـ Monthly Tracker **يشتغلان offline** — لا يحتاجان انترنت
- الشات بوت **يحتاج انترنت** — تأكد من WiFi أو هوت سبوت
- شغّل `docker-compose up` قبل الحفل — الـ API يكون جاهز في ثواني

---

## 👤 المسؤول

**Abdulrahman Metwally** — Tech Lead (Data & AI Track)
