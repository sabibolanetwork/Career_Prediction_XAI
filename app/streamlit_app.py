import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Path mappings relative to the app directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models/')

st.set_page_config(page_title="Career Prediction XAI", page_icon="🎓", layout="centered")

st.title("🎓 Undergraduate Career Prediction System")
st.write("Enter your academic background, skills, and interests below to generate an AI-driven career recommendation with full XAI explanations.")
st.markdown("---")

# --- Load Model, Unified Encoders, and Initialize SHAP ---
@st.cache_resource
def load_project_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_career_prediction_model.pkl")
    encoders_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    
    # Step 12.2: Create SHAP explainer instance using the loaded tree model
    explainer = shap.TreeExplainer(model)
    return model, encoders, explainer

try:
    model, encoders, explainer = load_project_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts. Verify files exist in the models directory. Error: {e}")
    st.stop()

# --- Create Interactive UI Selectors ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    age = st.selectbox("Age Range", encoders["Age"].classes_)
    level = st.selectbox("Level of Study", encoders["Level"].classes_)
    programme = st.selectbox("Programme", encoders["Programme"].classes_)
    cgpa = st.selectbox("CGPA Range", encoders["CGPA"].classes_)
    strongest_area = st.selectbox("Strongest Area", encoders["StrongestArea"].classes_)
    weakest_area = st.selectbox("Weakest Area", encoders["WeakestArea"].classes_)

with col2:
    programming_skill = st.selectbox("Programming Skill Level", encoders["ProgrammingSkill"].classes_)
    languages_known = st.selectbox("Programming Languages Known", encoders["LanguagesKnown"].classes_)
    projects_completed = st.selectbox("Completed IT Projects?", encoders["ProjectsCompleted"].classes_)
    internship = st.selectbox("Internship/SIWES Experience?", encoders["Internship"].classes_)
    technical_interest = st.selectbox("Technical Interest Focus", encoders["TechnicalInterest"].classes_)
    data_tools = st.selectbox("Familiarity with Data Tools", encoders["DataTools"].classes_)
    certifications = st.selectbox("Certifications Earned", encoders["Certifications"].classes_)

st.markdown("---")

# --- Form Inference & XAI Explanations Event ---
if st.button("Predict Optimal Career Path", use_container_width=True):
    # Package inputs into a matching structured DataFrame
    input_data = pd.DataFrame({
        "Gender": [gender], "Age": [age], "Level": [level], "Programme": [programme],
        "CGPA": [cgpa], "StrongestArea": [strongest_area], "WeakestArea": [weakest_area],
        "ProgrammingSkill": [programming_skill], "LanguagesKnown": [languages_known],
        "ProjectsCompleted": [projects_completed], "Internship": [internship],
        "TechnicalInterest": [technical_interest], "DataTools": [data_tools],
        "Certifications": [certifications]
    })
    
    # Encode the user input using the loaded encoders dictionary
    for col in input_data.columns:
        input_data[col] = encoders[col].transform(input_data[col].astype(str))
        
    # Predict the career path
    prediction = model.predict(input_data)[0]
    predicted_career = encoders["CareerPath"].inverse_transform([prediction])[0]
    
    # Calculate Prediction Confidence Score if supported by model
    confidence = model.predict_proba(input_data).max() * 100 if hasattr(model, "predict_proba") else None

    # Render results visually
    st.success(f"### 🎯 Predicted Career Path: **{predicted_career}**")
    if confidence is not None:
        st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
        
    st.markdown("---")
    
    # ==========================================================
    # STEP 12: SHAP INTERPRETABILITY EXECUTION
    # ==========================================================
    # Step 12.3: Generate SHAP values for the specific input vector
    shap_values = explainer.shap_values(input_data)
    
    st.subheader("💡 Explainable AI (XAI) Prediction Breakdown")
    st.write("The matrix and chart below show the features that most strongly influenced this specific prediction path.")

    # Step 12.4: Calculate multi-class local feature contributions
    predicted_class = prediction
    feature_names = input_data.columns
    
    # Handle both binary list outputs and raw 3D multiclass array shapes cleanly
    if isinstance(shap_values, list):
        shap_contributions = shap_values[predicted_class][0]
    elif len(shap_values.shape) == 3:
        shap_contributions = shap_values[0, :, predicted_class]
    else:
        shap_contributions = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": shap_contributions
    })
    
    explanation_df["Absolute Contribution"] = explanation_df["Contribution"].abs()
    explanation_df = explanation_df.sort_values(by="Absolute Contribution", ascending=False).reset_index(drop=True)
    
    # Display the top five contributing variables in a data frame
    st.write("**Top 5 Influential Factors Matrix:**")
    st.dataframe(explanation_df[["Feature", "Contribution"]].head(5), use_container_width=True)
    
    # Step 12.5: Display the clear text summary sentence
    top_feature = explanation_df.iloc[0]["Feature"]
    st.info(f"🔍 **Core Assessment Insight:** The most influential factor directing this specific prediction model recommendation was **{top_feature}**.")
    
    # Step 12.6: Generate local feature importance chart
    fig, ax = plt.subplots(figsize=(8, 4))
    top_five_df = explanation_df.head(5).sort_values(by="Absolute Contribution", ascending=True)
    
    ax.barh(top_five_df["Feature"], top_five_df["Absolute Contribution"], color="dodgerblue", edgecolor="black", height=0.5)
    ax.set_title("Top 5 Feature Contribution Magnitudes", fontsize=12, fontweight='bold')
    ax.set_xlabel("Absolute SHAP Impact Value")
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)
