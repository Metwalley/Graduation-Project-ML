# نظام التقييم النفسي للأطفال — AI Engine

**Graduation Project | Faculty of Computer Science**

نظام ذكاء اصطناعي متكامل للكشف المبكر عن الاضطرابات النمائية عند الأطفال ومتابعة تقدمهم.

---

## 🧠 ما الذي يفعله هذا النظام؟

يوفر هذا النظام **ثلاث قدرات أساسية** عبر API موحد:

| الميزة | الوصف | الدقة |
|---|---|---|
| **تشخيص التوحد** | نموذج XGBoost يحلل نتائج استبيان Q-Chat-10 | ~99% |
| **تشخيص ADHD** | نموذج XGBoost مع Proportional Normalization | ~76% |
| **تشخيص عسر القراءة** | نموذج Random Forest | — |
| **شات بوت تعليمي** | RAG محلي (Llama 3.2 + FAISS) بدون انترنت | — |
| **متتبع التقدم الشهري** | تسجيل وقياس تحسن الطفل شهرياً | — |

---

## 🚀 تشغيل النظام

### المتطلبات
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) مثبت وشغال

### التشغيل
```bash
docker-compose up --build
```

> ⚠️ **أول تشغيل فقط:** يحمّل موديل Llama 3.2 (~2GB). بعد كده هو محفوظ ومش هيحمّل تاني.

### التحقق من التشغيل
```
http://localhost:8000/health   → AI Engine status
http://localhost:8000/docs     → Swagger UI (توثيق تفاعلي لكل الـ endpoints)
```

---

## 📡 الـ API Endpoints

**Base URL:** `http://localhost:8000`

| Method | Endpoint | الوظيفة |
|---|---|---|
| `POST` | `/predict` | التشخيص الأولي |
| `POST` | `/monthly-tracker` | حساب التقدم الشهري |
| `GET` | `/monthly-tracker/questions/{disorder}` | جلب أسئلة التقييم |
| `POST` | `/chat` | الشات بوت التعليمي |
| `GET` | `/health` | فحص حالة الخدمة |

**التوثيق الكامل مع أمثلة:** راجع [`docs/TEAM_HANDOFF.md`](docs/TEAM_HANDOFF.md)

---

## 🗂️ هيكل المشروع

```
Graduation-Project-ML/
│
├── api/                        ← Unified FastAPI (المشغّل الرئيسي)
│   ├── main.py                 ← كل الـ endpoints (predict + tracker + chat)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── ml_models/              ← ملفات الموديلات (.joblib) — يقرأها الـ API مباشرة
│
├── chatbot_service/            ← RAG Chatbot (داخلي، يُشغّل عبر docker-compose)
│   ├── main.py
│   ├── rag_engine.py
│   ├── ollama_client.py
│   ├── knowledge_base.py
│   ├── safety_filter.py
│   ├── response_formatter.py
│   ├── config.py
│   └── Dockerfile
│
├── models/                     ← Training workspace (للـ ML engineer فقط)
│   ├── adhd/                   ← trainer + joblib files
│   ├── autism/                 ← trainer + joblib files
│   └── dyslexia/               ← trainer + joblib files
│
├── seed_data/                  ← البيانات المشتركة
│   ├── articles.json           ← مقالات: للشات بوت (RAG) + للعرض في الـ App
│   ├── exercises.json          ← تمارين وأنشطة للـ Backend يزرعها في MySQL
│   └── monthly_tracker.json    ← أسئلة التقييم الشهري
│
├── notebooks/                  ← Jupyter notebooks للتدريب والتحليل
├── docs/                       ← كل التوثيق
│   ├── TEAM_HANDOFF.md         ← دليل التكامل للتيم كله ← ابدأ بده
│   ├── PROJECT_BIBLE.md        ← المرجع التقني الشامل
│   ├── ARCHITECTURE.md         ← شرح معماري النظام
│   └── ...
│
└── docker-compose.yml          ← يشغّل كل حاجة بأمر واحد
```

---

## 🔗 التكامل مع باقي الفريق

**للـ Backend (Spring Boot):** كل الـ endpoints موثقة في `docs/TEAM_HANDOFF.md`.

**للـ Flutter:** راجع `docs/TEAM_HANDOFF.md` — قسم "كيف تتعاملوا مع كل Endpoint".

**للـ ML Engineer:** الموديلات والـ trainers في `models/`، الـ API جاهز في `api/`.

---

## 👤 المسؤول

**Abdulrahman Metwally** — Tech Lead (Data & AI Track)
