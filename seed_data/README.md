# `seed_data/` — Shared Data Files

Data files shared across services. **Do not modify without coordinating with the ML team.**

## Files

### `articles.json`
Educational articles in Arabic about ADHD, Autism, and Dyslexia.

Used for two things:
1. **RAG Chatbot** — Loaded into FAISS for semantic search at startup
2. **App Content** — Backend seeds into MySQL; Flutter displays as a browsable article feed

Fields: `id`, `category`, `title`, `summary`, `content`, `read_time`, `author`, `tags`

### `exercises.json`
Activities and exercises for children, grouped by disorder and type (physical / parent-child).

- Backend seeds into MySQL
- Flutter displays as activity cards filtered by the child's disorder

Fields: `id`, `category`, `title`, `type`, `difficulty`, `duration_minutes`, `age_range`, `description`, `goal`, `media_url`

### `monthly_tracker.json`
Question bank for the monthly progress assessment.

Used by `GET /monthly-tracker/questions/{disorder}` — Flutter fetches and renders these dynamically so questions are never hardcoded in the app.

## Category Values
All files use: `"ADHD"` | `"Autism"` | `"Dyslexia"`
