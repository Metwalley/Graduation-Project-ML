# `models/` — ML Training Workspace

Training scripts and model artifacts organized by disorder. **For ML use only — the API reads from `api/ml_models/`, not here.**

## Structure

```
models/
├── adhd/
│   ├── adhd_trainer.py
│   ├── prepare_adhd_data.py
│   ├── adhd_xgb_model_optimized.joblib
│   └── adhd_features.joblib
│
├── autism/
│   ├── autism_trainer.py
│   ├── autism_xgb_model.joblib
│   ├── autism_xgb_metrics.joblib
│   └── autism_features.joblib
│
└── dyslexia/
    ├── dyslexia_trainer.py
    ├── dyslexia_rf_model.joblib      ← production model
    ├── dyslexia_xgb_model.joblib     ← experimental
    ├── dyslexia_metrics.joblib
    └── dyslexia_features.joblib
```

## Model Summary

| Disorder | Algorithm | Accuracy | Notes |
|---|---|---|---|
| Autism | XGBoost | ~99% | Q-Chat-10 screening tool |
| ADHD | XGBoost | ~76% | Excellent for medical context |
| Dyslexia | Random Forest | — | Inverted scoring logic |

## ADHD Normalization — Important

The ADHD model was trained on large-scale questionnaire scores. The Flutter app sends small values (0, 1, 2). The API applies **Proportional Normalization** automatically before passing to the model.

Do not retrain without reviewing the normalization logic in `api/main.py`.
