# 🎓 Undergraduate Career Prediction System using Explainable AI (XAI)

An interactive, web-based Machine Learning application designed to predict optimal career pathways for undergraduate students based on their academic history, programming skills, tech interests, and certifications. 

This system integrates **Streamlit** for the frontend user interface, **Scikit-Learn** for predictive inference modeling, and **SHAP (SHapley Additive exPlanations)** to provide transparent, real-time Explainable AI (XAI) metrics explaining exactly *why* each career path was recommended.

---

## 📂 Project Architecture

```text
Career_Prediction_XAI/
│
├── app/
│   └── streamlit_app.py                      # Main UI and inference application code
│
├── models/
│   ├── best_career_prediction_model.pkl      # Trained predictive classification model
│   └── label_encoders.pkl                    # Reconstructed text categorical encoders
│
├── data/
│   └── raw/
│       └── career_questionnaire.csv          # Raw data asset used to re-fit categories
│
└── requirements.txt                          # Production library dependency manifest
