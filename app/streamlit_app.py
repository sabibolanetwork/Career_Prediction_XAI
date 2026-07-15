import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

MODELS_DIR = '/content/models/'

# Explicitly clear old Streamlit cache allocations to drop historical metrics
st.cache_resource.clear()

st.set_page_config(page_title="Career Prediction XAI", page_icon="🎓", layout="centered")

# --- Heading and New Welcoming Introduction Block ---
st.title("🎓 Undergraduate Career Prediction System")

# Structured, detailed, yet short introductory card
st.info("""
### 🌟 Welcome to the Career Prediction System!
This AI-driven diagnostic platform is designed to help you discover the optimal career path tailored to your unique academic profile, skill sets, and experiences.

* 📊 **What you will do:** Select your academic grades, technical capabilities, and internship background from the dropdown menus below.
* ⏱️ **Time required:** Less than 2 minutes.
* 🔒 **Data Privacy:** Your inputs are processed in real-time to generate predictions and explainable AI (XAI) charts. No personal data is stored or logged.
""")

st.markdown("---")

# --- Resource Loading Framework V2 ---
def load_resources():
    model = joblib.load(os.path.join(MODELS_DIR, "best_career_prediction_model_v2.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders_v2.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    
    # Extract underlying decision tree for SHAP
    base_estimator = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_estimator)
    return model, encoders, feature_columns, explainer

try:
    model, encoders, feature_columns, explainer = load_resources()
except Exception as e:
    st.error(f"Failed to load V2 model resources: {e}")
    st.stop()

# Interactive Selectors (Matches the template features exactly)
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    age = st.selectbox("Age Range", encoders["Age"].classes_)
    level = st.selectbox("Level of Study", encoders["Level"].classes_)
    programme = st.selectbox("Programme", encoders["Programme"].classes_)
    cgpa = st.selectbox("CGPA Range", encoders["CGPA"].classes_)
    strongest_area = st.selectbox("Strongest Area", encoders["StrongestArea"].classes_)
with col2:
    weakest_area = st.selectbox("Weakest Area", encoders["WeakestArea"].classes_)
    programming_skill = st.selectbox("Programming Skill Level", encoders["ProgrammingSkill"].classes_)
    languages_known = st.selectbox("Programming Languages Known", encoders["LanguagesKnown"].classes_)
    projects_completed = st.selectbox("Completed IT Projects?", encoders["ProjectsCompleted"].classes_)
    internship = st.selectbox("Internship/SIWES Experience?", encoders["Internship"].classes_)
    data_tools = st.selectbox("Familiarity with Data Tools", encoders["DataTools"].classes_)
    certifications = st.selectbox("Certifications Earned", encoders["Certifications"].classes_)

st.markdown("---")

if st.button("Predict Optimal Career Path", use_container_width=True):
    # 1. Capture Raw Text Input
    input_data = pd.DataFrame({
        "Gender": [gender], "Age": [age], "Level": [level], "Programme": [programme],
        "CGPA": [cgpa], "StrongestArea": [strongest_area], "WeakestArea": [weakest_area],
        "ProgrammingSkill": [programming_skill], "LanguagesKnown": [languages_known],
        "ProjectsCompleted": [projects_completed], "Internship": [internship],
        "DataTools": [data_tools], "Certifications": [certifications]
    })
    
    # 2. Map Text Directly to Separated Encoded Copy
    encoded_input = input_data.copy()
    for col in encoded_input.columns:
        encoded_input[col] = encoders[col].transform(encoded_input[col].astype(str))
        
    # 3. Force exact model matrix structural column alignment
    encoded_input = encoded_input[feature_columns]
    
    # 4. Generate Classification Output and Probability Distributions
    probabilities = model.predict_proba(encoded_input)[0]
    prediction = probabilities.argmax()
    predicted_career = encoders["CareerPath"].inverse_transform([prediction])[0]
    confidence = probabilities.max() * 100

    # Output Primary Layout
    st.success(f"### 🎯 Predicted Career Path: **{predicted_career}**")
    st.metric(label="Estimated Model Confidence", value=f"{confidence:.1f}%")
    
    st.markdown("---")
    
    # --- DIAGNOSTIC DEBUG ENGINE PANE ---
    st.markdown("### 🛠️ Systematic Pipeline Debugging Metrics")
    
    col_db1, col_db2 = st.columns(2)
    with col_db1:
        st.write("**1. Raw User Input (Widget State Check):**")
        st.dataframe(input_data, use_container_width=True)
    with col_db2:
        st.write("**2. Encoded Array Input (Matrix Check):**")
        st.dataframe(encoded_input, use_container_width=True)
        
    st.write("**3. Live Class Probability Shifts:**")
    prob_df = pd.DataFrame({
        "Career Track Option": encoders["CareerPath"].classes_,
        "Calculated Probability Score": probabilities
    }).sort_values(by="Calculated Probability Score", ascending=False).reset_index(drop=True)
    st.dataframe(prob_df, use_container_width=True)
    
    st.write("**4. Model Inspection Profile:**")
    st.caption(f"Loaded Pipeline Class: `{type(model).__name__}` | Base Target Alignment: `{list(feature_columns)}`")
    
    st.markdown("---")
    
    # --- SHAP Interpretation ---
    try:
        shap_values = explainer.shap_values(encoded_input)
        st.subheader("💡 Explainable AI (XAI) Prediction Breakdown")
        
        if isinstance(shap_values, list):
            shap_contributions = shap_values[prediction][0]
        elif len(shap_values.shape) == 3:
            shap_contributions = shap_values[0, :, prediction]
        else:
            shap_contributions = shap_values[0]

        explanation_df = pd.DataFrame({"Feature": encoded_input.columns, "Contribution": shap_contributions})
        explanation_df["Absolute Contribution"] = explanation_df["Contribution"].abs()
        explanation_df = explanation_df.sort_values(by="Absolute Contribution", ascending=False).reset_index(drop=True)
        
        top_features = explanation_df.head(3)
        feature_text = ", ".join(top_features["Feature"].tolist())
        st.write(f"The most influential factors in this recommendation were **{feature_text}**.")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        top_five_df = explanation_df.head(5).sort_values(by="Absolute Contribution", ascending=True)
        ax.barh(top_five_df["Feature"], top_five_df["Absolute Contribution"], color="dodgerblue", edgecolor="black", height=0.5)
        ax.set_title("Top 5 Feature Contribution Magnitudes")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info(
            """
            **How to read this explanation**
            Each bar shows how a student characteristic influenced the recommended career. 
            Positive contributions pushed the model toward the displayed career, while negative contributions pushed it away. 
            Longer bars indicate a stronger influence. The explanation describes the model's reasoning; it does not prove that a feature causes a particular career outcome.
            """
        )
    except Exception as e:
        st.warning(f"SHAP diagnostics pending interface sync. Error: {e}")
