import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import io

warnings.filterwarnings('ignore')

# --- Page Config and CSS ---
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app container */
    .stApp {
        background-color: #e6f0ff; /* Lighter, professional blue */
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-top: 1rem;
        color: #2c3e50;
    }
    
    /* Dynamic Heartbeat Animation */
    @keyframes heartbeat {
        0% { transform: scale(1); }
        25% { transform: scale(1.1); }
        50% { transform: scale(1.2); }
        75% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .heart {
        animation: heartbeat 2s infinite;
        color: #e74c3c;
        font-size: 4rem;
        display: inline-block;
        margin-right: 1rem;
        filter: drop-shadow(0 0 10px rgba(231, 76, 60, 0.5));
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.5rem;
        color: #0b1f3a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 2rem;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1f4068, #355685);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .metric-card p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .info-box {
        background-color: #3572ad;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: none;
    }
    
    .warning-box {
        background-color: #e74c3c;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* The color of this box is now blue */
    .success-box {
        background-color: #3498db;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0d284a;
    }
    
    .css-1d391kg .css-1outpf7 {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Form styling */
    .stForm {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: #3498db;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 12px 30px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: #2980b9;
    }
    
    /* Tab styling (used for st.selectbox as a navigation bar) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3498db;
        color: white !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stNumberInput input, .stSelectbox select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* Section headers */
    .section-header {
        background: #34495e;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Results styling */
    .result-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* Data quality indicators */
    .quality-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .high-quality {
        background: #00b894;
        color: white;
    }
    
    .medium-quality {
        background: #fdcb6e;
        color: #2d3436;
    }
    
    .low-quality {
        background: #e17055;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- App Logic and Functions ---

# The scaler expects only 8 features
SCALER_FEATURES = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak']

# Load the saved machine learning pipeline components
@st.cache_resource
def load_model():
    """Loads the model and scaler components from the pickled file."""
    try:
        # NOTE: This file must exist in the same directory for the code to run locally.
        with open('heart_model_pipeline_v2.pkl', 'rb') as file:
            pipeline_components = pickle.load(file)
        return pipeline_components['model'], pipeline_components['scaler']
    except FileNotFoundError:
        # Placeholder/Mock data for execution environment where the file doesn't exist
        st.warning("❌ Model file 'heart_model_pipeline_v2.pkl' not found. Using mock model/scaler for demonstration.")
        
        # Mock Scaler for demonstration (only transforms data structure, not actual scaling)
        class MockScaler:
            def transform(self, data):
                return data.values[:, :len(SCALER_FEATURES)] 

        # Mock Model for demonstration
        class MockModel:
            def predict(self, features):
                # Simple mock prediction: result is 1 if Age * Cholesterol is high (first and fourth features of SCALER_FEATURES)
                try:
                    # Simple heuristic: high age (feature 0) and high cholesterol (feature 3) is high risk (1)
                    return np.array([1 if f[0] > 50 and f[3] > 200 else 0 for f in features])
                except IndexError:
                    return np.array([0] * features.shape[0])

            def predict_proba(self, features):
                predictions = self.predict(features)
                probs = np.array([[0.2, 0.8] if p == 1 else [0.8, 0.2] for p in predictions])
                return probs
            
        return MockModel(), MockScaler()
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        st.stop()

model, scaler = load_model()

def normalize_column_names(df):
    """Normalize column names to handle different variations"""
    df_normalized = df.copy()
    if 'ChestPainType' in df_normalized.columns and 'ChestPain' not in df_normalized.columns:
        df_normalized['ChestPain'] = df_normalized['ChestPainType']
        df_normalized = df_normalized.drop('ChestPainType', axis=1)
    return df_normalized

def prepare_features_for_scaler(df):
    """Prepare only the 8 features that scaler expects"""
    df_copy = normalize_column_names(df)
    
    if 'Sex' in df_copy.columns:
        df_copy['Sex'] = df_copy['Sex'].astype(str).str.upper().map({'M': 1, 'F': 0}).fillna(df_copy['Sex'])
        df_copy['Sex'] = pd.to_numeric(df_copy['Sex'], errors='coerce').fillna(0).astype(int)
    
    if 'ExerciseAngina' in df_copy.columns:
        df_copy['ExerciseAngina'] = df_copy['ExerciseAngina'].astype(str).str.upper().map({'Y': 1, 'N': 0}).fillna(df_copy['ExerciseAngina'])
        df_copy['ExerciseAngina'] = pd.to_numeric(df_copy['ExerciseAngina'], errors='coerce').fillna(0).astype(int)

    # Ensure all scaler features exist, filling missing numerical ones with 0 before selection
    for feature in SCALER_FEATURES:
        if feature not in df_copy.columns:
            df_copy[feature] = 0
        elif df_copy[feature].dtype == object: # Handle cases where numeric features are read as object/string
            df_copy[feature] = pd.to_numeric(df_copy[feature], errors='coerce').fillna(df_copy[feature].median() if len(df_copy) > 1 and df_copy[feature].notnull().any() else 0)

    return df_copy[SCALER_FEATURES]

def prepare_features_for_model(df):
    """Prepare all 18 features that model expects (encoded + numerical)"""
    df_copy = normalize_column_names(df)
    
    # 1. Sex encoding
    if 'Sex' in df_copy.columns:
        df_copy['Sex'] = df_copy['Sex'].astype(str).str.upper().map({'M': 1, 'F': 0}).fillna(df_copy['Sex'])
        df_copy['Sex'] = pd.to_numeric(df_copy['Sex'], errors='coerce').fillna(0).astype(int)
    
    # 2. ChestPain encoding
    if 'ChestPain' in df_copy.columns:
        chest_pain_dummies = pd.get_dummies(df_copy['ChestPain'], prefix='ChestPain')
        df_copy = pd.concat([df_copy, chest_pain_dummies], axis=1)
        df_copy = df_copy.drop('ChestPain', axis=1)
    
    for cp_type in ['ChestPain_ATA', 'ChestPain_ASY', 'ChestPain_NAP', 'ChestPain_TA']:
        if cp_type not in df_copy.columns:
            df_copy[cp_type] = 0
    
    # 3. RestingECG encoding
    if 'RestingECG' in df_copy.columns:
        resting_ecg_dummies = pd.get_dummies(df_copy['RestingECG'], prefix='RestingECG')
        df_copy = pd.concat([df_copy, resting_ecg_dummies], axis=1)
        df_copy = df_copy.drop('RestingECG', axis=1)
    
    for ecg_type in ['RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST']:
        if ecg_type not in df_copy.columns:
            df_copy[ecg_type] = 0
    
    # 4. ExerciseAngina encoding
    if 'ExerciseAngina' in df_copy.columns:
        df_copy['ExerciseAngina'] = df_copy['ExerciseAngina'].astype(str).str.upper().map({'Y': 1, 'N': 0}).fillna(df_copy['ExerciseAngina'])
        df_copy['ExerciseAngina'] = pd.to_numeric(df_copy['ExerciseAngina'], errors='coerce').fillna(0).astype(int)
    
    # 5. ST_Slope encoding
    if 'ST_Slope' in df_copy.columns:
        st_slope_dummies = pd.get_dummies(df_copy['ST_Slope'], prefix='ST_Slope')
        df_copy = pd.concat([df_copy, st_slope_dummies], axis=1)
        df_copy = df_copy.drop('ST_Slope', axis=1)
    
    for slope_type in ['ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']:
        if slope_type not in df_copy.columns:
            df_copy[slope_type] = 0
    
    model_features = [
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'ExerciseAngina',
        'ChestPain_ATA', 'ChestPain_ASY', 'ChestPain_NAP', 'ChestPain_TA',
        'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
        'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    for col in model_features:
        if col not in df_copy.columns:
            df_copy[col] = 0
        elif col in SCALER_FEATURES and df_copy[col].dtype == object:
             df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(df_copy[col].median() if len(df_copy) > 1 and df_copy[col].notnull().any() else 0)

    return df_copy[model_features]

def make_prediction(input_df):
    """Make predictions with correct feature handling"""
    try:
        # Prepare and scale the numerical features
        scaler_data = prepare_features_for_scaler(input_df)
        scaled_data = scaler.transform(scaler_data)
        
        # Prepare the full feature set (unscaled numerical + encoded categorical)
        model_data_df = prepare_features_for_model(input_df)
        
        # Replace the unscaled numerical columns with the scaled versions
        for i, feature in enumerate(SCALER_FEATURES):
            if feature in model_data_df.columns:
                # Use the scaled data derived from the input's SCALER_FEATURES (order is maintained by prepare_features_for_scaler)
                model_data_df[feature] = scaled_data[:, i]
        
        final_features = model_data_df.values
        
        predictions = model.predict(final_features)
        probabilities = model.predict_proba(final_features)
        
        return predictions, probabilities
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None

def get_risk_interpretation(risk_prob):
    """Get risk interpretation based on probability"""
    if risk_prob >= 0.8:
        return "🔴 VERY HIGH RISK", "Immediate medical attention recommended"
    elif risk_prob >= 0.6:
        return "🟠 HIGH RISK", "Consult cardiologist soon"
    elif risk_prob >= 0.4:
        return "🟡 MODERATE RISK", "Regular monitoring advised"
    elif risk_prob >= 0.2:
        return "🟢 LOW RISK", "Maintain healthy lifestyle"
    else:
        return "✅ VERY LOW RISK", "Continue current health practices"

# --- Sidebar Content ---
with st.sidebar:
    st.markdown('<div class="section-header">🏥 Model Information</div>', unsafe_allow_html=True)
    st.markdown("""
    *🤖 AI Model Details:*
    - *Algorithm:* Random Forest Classifier
    - *Features:* 18 clinical parameters
    - *Accuracy:* 92.3% on test data
    - *Training Data:* 918 patient records
    - *Status:* ✅ Active & Validated
    """)
    
    st.markdown('<div class="section-header">📋 Required Parameters</div>', unsafe_allow_html=True)
    st.markdown("""
    *Primary Metrics:*
    - Age, Gender, Blood Pressure
    - Cholesterol, Fasting Blood Sugar
    - Maximum Heart Rate, ST Depression
    
    *Clinical Categories:*
    - Chest Pain Type, Resting ECG
    - Exercise Angina, ST Slope
    """)
    
    st.markdown('<div class="section-header">⚕ Medical Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; font-size: 0.9rem;">
    This tool is for educational and screening purposes only. 
    Always consult qualified healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)

# --- Main App Logic ---
st.markdown('<h1 class="main-header"><span class="heart">🫀</span> Heart Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box">🔬 Advanced AI-powered cardiovascular risk assessment and analysis system for healthcare professionals</div>', unsafe_allow_html=True)

# Tabs are now implemented using st.selectbox for stable navigation
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "🏠 Home"

st.session_state.selected_tab = st.selectbox(
    "Navigation", 
    ["🏠 Home", "🩺 Manual Assessment", "📊 Bulk Analysis", "📈 Visualization"],
    index=["🏠 Home", "🩺 Manual Assessment", "📊 Bulk Analysis", "📈 Visualization"].index(st.session_state.selected_tab),
    label_visibility="collapsed"
)

# Content based on selected tab
if st.session_state.selected_tab == "🏠 Home":
    st.markdown('<h2 class="sub-header">🏥 Welcome to Heart Disease Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🔬 Advanced Cardiovascular Risk Assessment</h3>
            <p>CardioPredict is a state-of-the-art AI system designed to assist healthcare professionals 
            in rapid cardiovascular risk evaluation. Our machine learning model analyzes 18 key health 
            parameters to provide accurate risk predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>✨ Key Features</h4>
            <ul>
                <li><strong>Individual Assessment:</strong> Real-time analysis of patient data</li>
                <li><strong>Batch Processing:</strong> Analyze multiple patients simultaneously</li>
                <li><strong>Visual Analytics:</strong> Comprehensive charts and insights</li>
                <li><strong>Risk Stratification:</strong> Clear risk categories and recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>92.3%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>918</h3>
            <p>Training Records</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>18</h3>
            <p>Clinical Parameters</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.selected_tab == "🩺 Manual Assessment":
    st.markdown('<h2 class="sub-header">🩺 Individual Patient Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📝 Patient Information</div>', unsafe_allow_html=True)
    
    with st.form("manual_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("👤 Demographics**")
            age = st.number_input("🎂 Age (years)", min_value=1, max_value=120, value=None, placeholder="Enter age", help="Patient's age in years")
            sex = st.selectbox("⚥ Gender", options=['M', 'F'], index=None, 
                              format_func=lambda x: "Select Gender" if x is None else "👨 Male" if x == 'M' else "👩 Female")
            
            st.markdown("💓 Cardiovascular**")
            resting_bp = st.number_input("🩸 Resting BP (mmHg)", min_value=80, max_value=200, value=None, placeholder="Enter BP",
                                        help="Resting blood pressure in mmHg")
            cholesterol = st.number_input("🧪 Cholesterol (mg/dl)", min_value=0, max_value=600, value=None, placeholder="Enter cholesterol",
                                         help="Serum cholesterol level")
        
        with col2:
            st.markdown("🔬 Clinical Tests**")
            fasting_bs = st.selectbox("🍽 Fasting Blood Sugar > 120 mg/dl", options=[0, 1], index=None,
                                     format_func=lambda x: "Select Option" if x is None else "✅ Yes" if x == 1 else "❌ No")
            max_hr = st.number_input("❤ Max Heart Rate (bpm)", min_value=60, max_value=220, value=None, placeholder="Enter max HR",
                                    help="Maximum heart rate achieved during exercise")
            
            st.markdown("🏃 Exercise Response**")
            exercise_angina = st.selectbox("💔 Exercise Induced Angina", options=['Y', 'N'], index=None,
                                          format_func=lambda x: "Select Option" if x is None else "✅ Yes" if x == 'Y' else "❌ No")
            oldpeak = st.number_input("📉 ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=None, placeholder="Enter ST depression", step=0.1,
                                     help="ST depression induced by exercise")
        
        with col3:
            st.markdown("📋 Clinical Categories**")
            chest_pain_options = {
                'ATA': '⚡ Atypical Angina',
                'NAP': '🔸 Non-Anginal Pain', 
                'ASY': '🔹 Asymptomatic',
                'TA': '💔 Typical Angina'
            }
            chest_pain = st.selectbox("💔 Chest Pain Type", 
                                     options=list(chest_pain_options.keys()), index=None,
                                     format_func=lambda x: "Select Type" if x is None else chest_pain_options[x])
            
            ecg_options = {
                'Normal': '✅ Normal',
                'ST': '📈 ST-T Wave Abnormality',
                'LVH': '🫀 Left Ventricular Hypertrophy'
            }
            resting_ecg = st.selectbox("📊 Resting ECG", 
                                      options=list(ecg_options.keys()), index=None,
                                      format_func=lambda x: "Select Type" if x is None else ecg_options[x])
            
            slope_options = {
                'Up': '📈 Upsloping',
                'Flat': '➡ Flat',
                'Down': '📉 Downsloping'
            }
            st_slope = st.selectbox("📈 ST Slope", 
                                   options=list(slope_options.keys()), index=None,
                                   format_func=lambda x: "Select Type" if x is None else slope_options[x])
    
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔍 Analyze Risk Profile", use_container_width=True, type="primary")

    if submitted:
        # Validation - check if all fields are filled (None values)
        if any(v is None for v in [age, sex, fasting_bs, exercise_angina, oldpeak, resting_bp, cholesterol, max_hr, chest_pain, resting_ecg, st_slope]):
            st.error("⚠ Please fill all fields before submitting.")
        else:
            input_data = pd.DataFrame({
                'Age': [age], 'Sex': [sex], 'ChestPain': [chest_pain], 'RestingBP': [resting_bp],
                'Cholesterol': [cholesterol], 'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg],
                'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina], 'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]
            })
            
            prediction, prediction_proba = make_prediction(input_data)
            
            if prediction is not None:
                result = {
                    'prediction': prediction[0],
                    'prob_disease': prediction_proba[0][1],
                    'prob_healthy': prediction_proba[0][0],
                    'risk_percentage': prediction_proba[0][1] * 100
                }
                
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                risk_level, recommendation = get_risk_interpretation(result['prob_disease'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_color = "background-color: #e74c3c;" if result['prediction'] == 1 else "background-color: #27ae60;"
                    risk_text_parts = risk_level.split()
                    # Only show the risk level, not the icon
                    risk_text = " ".join(risk_text_parts[1:]) 
                    st.markdown(f"""
                    <div class="metric-card" style="{risk_color}">
                        <h3>{risk_text}</h3>
                        <p>Risk Assessment</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    disease_pct = f"{result['risk_percentage']:.1f}"
                    st.markdown(f"""
                    <div class="metric-card" style="background-color: #e67e22;">
                        <h3>{disease_pct}&#37;</h3>
                        <p>Disease Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    healthy_pct = f"{(1-result['prob_disease'])*100:.1f}"
                    st.markdown(f"""
                    <div class="metric-card" style="background-color: #3498db;">
                        <h3>{healthy_pct}&#37;</h3>
                        <p>Healthy Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📋 Clinical Interpretation")
                
                if result['prob_disease'] >= 0.6:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>{risk_level}</h4>
                        <p><strong>Recommendation:</strong> {recommendation}</p>
                        <p><strong>Next Steps:</strong> Comprehensive cardiovascular evaluation recommended. 
                        Consider additional diagnostic tests such as stress testing, echocardiogram, or coronary angiography.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>{risk_level}</h4>
                        <p><strong>Recommendation:</strong> {recommendation}</p>
                        <p><strong>Next Steps:</strong> Continue regular health maintenance. 
                        Monitor cardiovascular risk factors and maintain healthy lifestyle practices.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.selected_tab == "📊 Bulk Analysis":
    st.markdown('<h2 class="sub-header">📊 Bulk Patient Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Upload a CSV file containing multiple patient records for batch processing and comprehensive analysis.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📁 Upload Patient Dataset", type="csv", 
                                    help="Upload a CSV file with patient data following the required format")
    
    if uploaded_file:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
                st.dataframe(df_uploaded.head(10), use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-header">📈 Dataset Statistics</div>', unsafe_allow_html=True)
                
                total_patients = len(df_uploaded)
                total_features = len(df_uploaded.columns)
                missing_data = df_uploaded.isnull().sum().sum()
                data_quality = (1 - missing_data/(total_patients * total_features)) * 100 if total_patients * total_features > 0 else 100
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_patients}</h3>
                    <p>Total Patients</p>
                </div>
                """, unsafe_allow_html=True)
                
                quality_class = "high-quality" if data_quality > 95 else "medium-quality" if data_quality > 85 else "low-quality"
                quality_text = f"{data_quality:.1f}&#37;"
                st.markdown(f"""
                <div class="metric-card">
                    <h3><span class="quality-indicator {quality_class}">{quality_text}</span></h3>
                    <p>Data Quality</p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔬 Generate Predictions for All Patients", use_container_width=True, type="primary"):
                with st.spinner("🔄 Processing patient data... This may take a moment."):
                    predictions, probabilities = make_prediction(df_uploaded)
                    
                    if predictions is not None:
                        df_uploaded['Prediction'] = predictions
                        df_uploaded['Risk_Probability'] = [p[1] for p in probabilities]
                        df_uploaded['Healthy_Probability'] = [p[0] for p in probabilities]
                        df_uploaded['Risk_Category'] = df_uploaded['Risk_Probability'].apply(
                            lambda x: 'Very High' if x > 0.8 else 'High' if x > 0.6 else 'Moderate' if x > 0.4 else 'Low' if x > 0.2 else 'Very Low'
                        )
                        df_uploaded['Risk_Level'] = df_uploaded['Prediction'].apply(
                            lambda x: 'High Risk' if x == 1 else 'Low Risk'
                        )
                        
                        st.session_state['df_results'] = df_uploaded.copy()
                        
                        st.markdown('<div class="section-header">📊 Analysis Summary</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        high_risk = sum(df_uploaded['Prediction'] == 1)
                        high_risk_pct = f"{(high_risk/len(df_uploaded))*100:.1f}"
                        
                        low_risk = sum(df_uploaded['Prediction'] == 0)
                        low_risk_pct = f"{(low_risk/len(df_uploaded))*100:.1f}"
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card" style="background-color: #e74c3c">
                                <h3>{high_risk}</h3>
                                <p>High Risk ({high_risk_pct}&#37;)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card" style="background-color: #27ae60">
                                <h3>{low_risk}</h3>
                                <p>Low Risk ({low_risk_pct}&#37;)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            avg_risk = f"{df_uploaded['Risk_Probability'].mean()*100:.1f}"
                            st.markdown(f"""
                            <div class="metric-card" style="background-color: #e67e22">
                                <h3>{avg_risk}&#37;</h3>
                                <p>Average Risk</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            max_risk = f"{df_uploaded['Risk_Probability'].max()*100:.1f}"
                            st.markdown(f"""
                            <div class="metric-card" style="background-color: #9b59b6">
                                <h3>{max_risk}&#37;</h3>
                                <p>Highest Risk</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        risk_counts = df_uploaded['Risk_Category'].value_counts()
                        # Use all defined colors
                        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
                        
                        # Use all defined categories for consistent order/color mapping
                        category_order = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
                        risk_counts = risk_counts.reindex(category_order).fillna(0)
                        risk_counts = risk_counts[risk_counts > 0] # Filter out zero counts
                        
                        wedges, texts, autotexts = ax.pie(risk_counts.values, labels=risk_counts.index, 
                                                         autopct='%1.1f%%', 
                                                         colors=colors[:len(risk_counts)], 
                                                         startangle=90,
                                                         explode=[0.05]*len(risk_counts), shadow=True)
                        ax.set_title('Risk Category Distribution', fontsize=16, fontweight='bold', pad=20)
                        
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(12)
                            autotext.set_weight('bold')
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
                        # Display all columns for detailed results
                        st.dataframe(df_uploaded, use_container_width=True)
                        
                        # Download Button
                        @st.cache_data
                        def convert_df(df):
                            return df.to_csv(index=False).encode('utf-8')
                        
                        csv = convert_df(df_uploaded)

                        st.download_button(
                            label="💾 Download Complete Analysis",
                            data=csv,
                            file_name=f'cardiopredict_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    else:
                        st.error("❌ Error processing the dataset. Please check the data format.")
        
        except Exception as e:
            st.error(f"❌ Error reading the file: {str(e)}")
            st.info("💡 Please ensure your CSV file contains the required columns with proper formatting.")

elif st.session_state.selected_tab == "📈 Visualization":
    st.markdown('<h2 class="sub-header">📈 Advanced Visualization Home</h2>', unsafe_allow_html=True)
    
    if 'df_results' in st.session_state and not st.session_state['df_results'].empty:
        df_viz = st.session_state['df_results'].copy()
        df_viz['Prediction_Label'] = df_viz['Prediction'].apply(lambda x: 'Heart Disease' if x == 1 else 'Healthy')
        
        if 'Sex' in df_viz.columns:
            # Handle both string ('M'/'F') and numeric (1/0) based on how the file was read
            df_viz['Gender'] = df_viz['Sex'].map({1: 'Male', 0: 'Female', 'M': 'Male', 'F': 'Female'}).fillna('Unknown')
        
        st.markdown('<div class="section-header">📊 Choose Analysis Type</div>', unsafe_allow_html=True)
        
        viz_options = {
            "Executive Summary": "📊 High-level overview and key insights",
            "Demographic Analysis": "👥 Age and gender distribution analysis",
            "Clinical Parameters": "🩺 Blood pressure, cholesterol, and other metrics",
            "Risk Profiling": "⚠ Risk probability and category analysis",
            "Correlation Matrix": "🔥 Feature relationships and dependencies",
            "Predictive Insights": "🔬 Model confidence and feature importance"
        }
        
        selected_viz = st.selectbox(
            "Select Analysis:", 
            options=list(viz_options.keys()),
            format_func=lambda x: viz_options[x]
        )
        
        if selected_viz == "Executive Summary":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_patients = len(df_viz)
                st.markdown(f"""
                <div class="metric-card" style="background-color: #3498db;">
                    <h3>👥 {total_patients}</h3>
                    <p>Total Patients</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                high_risk_count = sum(df_viz['Prediction'] == 1)
                high_risk_pct = f"{(high_risk_count/total_patients)*100:.1f}"
                st.markdown(f"""
                <div class="metric-card" style="background-color: #e74c3c;">
                    <h3>🔴 {high_risk_pct}&#37;</h3>
                    <p>High Risk Rate</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                avg_age = f"{df_viz['Age'].mean():.1f}"
                st.markdown(f"""
                <div class="metric-card" style="background-color: #e67e22;">
                    <h3>📊 {avg_age}</h3>
                    <p>Average Age</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                avg_risk = f"{df_viz['Risk_Probability'].mean()*100:.1f}"
                st.markdown(f"""
                <div class="metric-card" style="background-color: #9b59b6;">
                    <h3>⚠ {avg_risk}&#37;</h3>
                    <p>Average Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                risk_counts = df_viz['Prediction_Label'].value_counts()
                colors = ['#27ae60', '#e74c3c']
                wedges, texts, autotexts = ax1.pie(risk_counts.values, labels=risk_counts.index, 
                                                  autopct='%1.1f%%', colors=colors, startangle=90,
                                                  explode=(0.1, 0.1), shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
                ax1.set_title('Overall Risk Distribution', fontsize=16, fontweight='bold', pad=20)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(14)
                    autotext.set_weight('bold')
                st.pyplot(fig1)
                plt.close()
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                age_bins = pd.cut(df_viz['Age'], bins=[0, 30, 40, 50, 60, 70, 100], 
                                 labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+'])
                age_risk = pd.crosstab(age_bins, df_viz['Prediction_Label'])
                age_risk.plot(kind='bar', ax=ax2, color=['#3498db', '#e67e22'], width=0.8)
                ax2.set_title('Risk Distribution by Age Groups', fontsize=16, fontweight='bold', pad=20)
                ax2.set_xlabel('Age Groups', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
                ax2.legend(title='Diagnosis', frameon=True, fancybox=True, shadow=True)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
        
        elif selected_viz == "Demographic Analysis":
            st.markdown("### 👥 Comprehensive Demographic Analysis")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                sns.violinplot(data=df_viz, x='Prediction_Label', y='Age', ax=ax1, palette=['#3498db', '#e74c3c'], inner='box')
                ax1.set_title('Age Distribution by Heart Disease Status', fontsize=16, fontweight='bold', pad=20)
                ax1.set_xlabel('Diagnosis', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Age (years)', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()
            with col2:
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                sns.histplot(data=df_viz, x='Age', hue='Prediction_Label', multiple='dodge', ax=ax2, palette=['#3498db', '#e74c3c'], alpha=0.7, bins=20)
                ax2.set_title('Age Frequency Distribution', fontsize=16, fontweight='bold', pad=20)
                ax2.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax2.legend(title='Diagnosis', frameon=True)
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            
            if 'Gender' in df_viz.columns and len(df_viz['Gender'].unique()) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    fig3, ax3 = plt.subplots(figsize=(12, 8))
                    gender_counts = pd.crosstab(df_viz['Gender'], df_viz['Prediction_Label'])
                    gender_counts.plot(kind='bar', ax=ax3, color=['#3498db', '#e67e22'], width=0.6)
                    ax3.set_title('Gender Distribution by Heart Disease Status', fontsize=16, fontweight='bold', pad=20)
                    ax3.set_xlabel('Gender', fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
                    ax3.legend(title='Diagnosis', frameon=True)
                    ax3.tick_params(axis='x', rotation=0)
                    ax3.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close()
                with col2:
                    fig4, ax4 = plt.subplots(figsize=(12, 8))
                    sns.violinplot(data=df_viz, x='Gender', y='Risk_Probability', ax=ax4, palette=['#e74c3c', '#3498db'])
                    ax4.set_title('Risk Probability Distribution by Gender', fontsize=16, fontweight='bold', pad=20)
                    ax4.set_ylabel('Risk Probability', fontsize=12, fontweight='bold')
                    ax4.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close()
            st.markdown("### 📊 Demographic Statistics")
            demo_stats = df_viz.groupby('Prediction_Label')['Age'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)
            st.dataframe(demo_stats, use_container_width=True)
        
        elif selected_viz == "Clinical Parameters":
            st.markdown("### 🩺 Clinical Parameters Analysis")
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('Clinical Metrics Comprehensive Analysis', fontsize=24, fontweight='bold', y=0.98)
            sns.violinplot(data=df_viz, x='Prediction_Label', y='RestingBP', ax=axes[0,0], palette=['#3498db', '#e74c3c'], inner='box')
            axes[0,0].set_title('Resting Blood Pressure Distribution', fontweight='bold', fontsize=16, pad=15)
            axes[0,0].set_xlabel('Diagnosis', fontsize=14, fontweight='bold')
            axes[0,0].set_ylabel('Resting BP (mmHg)', fontsize=14, fontweight='bold')
            axes[0,0].grid(True, alpha=0.3)
            sns.violinplot(data=df_viz, x='Prediction_Label', y='Cholesterol', ax=axes[0,1], palette=['#27ae60', '#e67e22'], inner='box')
            axes[0,1].set_title('Cholesterol Level Distribution', fontweight='bold', fontsize=16, pad=15)
            axes[0,1].set_xlabel('Diagnosis', fontsize=14, fontweight='bold')
            axes[0,1].set_ylabel('Cholesterol (mg/dl)', fontsize=14, fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
            sns.violinplot(data=df_viz, x='Prediction_Label', y='MaxHR', ax=axes[1,0], palette=['#9b59b6', '#f39c12'], inner='box')
            axes[1,0].set_title('Maximum Heart Rate Distribution', fontweight='bold', fontsize=16, pad=15)
            axes[1,0].set_xlabel('Diagnosis', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('Max Heart Rate (bpm)', fontsize=14, fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
            sns.violinplot(data=df_viz, x='Prediction_Label', y='Oldpeak', ax=axes[1,1], palette=['#1abc9c', '#e74c3c'], inner='box')
            axes[1,1].set_title('ST Depression (Oldpeak) Distribution', fontweight='bold', fontsize=16, pad=15)
            axes[1,1].set_xlabel('Diagnosis', fontsize=14, fontweight='bold')
            axes[1,1].set_ylabel('ST Depression', fontsize=14, fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("### 📊 Clinical Parameters Statistics")
            clinical_cols = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            available_clinical = [col for col in clinical_cols if col in df_viz.columns]
            if available_clinical:
                clinical_stats = df_viz.groupby('Prediction_Label')[available_clinical].agg(['mean', 'median', 'std']).round(2)
                st.dataframe(clinical_stats, use_container_width=True)
        
        elif selected_viz == "Risk Profiling":
            st.markdown("### ⚠ Comprehensive Risk Analysis")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                n, bins, patches = ax1.hist(df_viz['Risk_Probability'], bins=30, alpha=0.8, edgecolor='black', linewidth=1.2)
                for i, p in enumerate(patches):
                    # Color coding for risk bins
                    if bins[i] < 0.2: p.set_facecolor('#27ae60')
                    elif bins[i] < 0.4: p.set_facecolor('#f39c12')
                    elif bins[i] < 0.6: p.set_facecolor('#e67e22')
                    else: p.set_facecolor('#e74c3c')
                    p.set_alpha(0.8)
                mean_risk = df_viz['Risk_Probability'].mean()
                median_risk = df_viz['Risk_Probability'].median()
                ax1.axvline(mean_risk, color='red', linestyle='--', linewidth=3, label=f'Mean: {mean_risk:.3f}')
                ax1.axvline(median_risk, color='blue', linestyle='--', linewidth=3, label=f'Median: {median_risk:.3f}')
                ax1.set_title('Risk Probability Distribution', fontsize=16, fontweight='bold', pad=20)
                ax1.set_xlabel('Risk Probability', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()
            with col2:
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                risk_counts = df_viz['Risk_Category'].value_counts()
                colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
                category_order = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
                risk_counts = risk_counts.reindex(category_order).fillna(0)
                risk_counts = risk_counts[risk_counts > 0]
                
                wedges, texts, autotexts = ax2.pie(risk_counts.values, labels=risk_counts.index, 
                                                  autopct='%1.1f%%', colors=colors[:len(risk_counts)], 
                                                  startangle=90, explode=[0.05]*len(risk_counts), shadow=True,
                                                  textprops={'fontsize': 11, 'fontweight': 'bold'})
                ax2.set_title('Risk Category Distribution', fontsize=16, fontweight='bold', pad=20)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(12)
                    autotext.set_weight('bold')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            
            fig3, ax3 = plt.subplots(figsize=(15, 8))
            scatter = ax3.scatter(df_viz['Age'], df_viz['Risk_Probability'], 
                                 c=df_viz['Risk_Probability'], cmap='RdYlBu_r', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax3.set_title('Risk Probability vs Age Analysis', fontsize=18, fontweight='bold', pad=20)
            ax3.set_xlabel('Age (years)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Risk Probability', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Risk Probability', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
        
        elif selected_viz == "Correlation Matrix":
            st.markdown("### 🔥 Feature Correlation Analysis")
            numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Risk_Probability']
            available_cols = [col for col in numeric_cols if col in df_viz.columns]
            if len(available_cols) >= 2:
                corr_matrix = df_viz[available_cols].corr()
                fig, ax = plt.subplots(figsize=(14, 12))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, ax=ax, 
                                    cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}, fmt='.3f', linewidths=0.5,
                                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
                ax.set_title('Feature Correlation Matrix', fontsize=20, fontweight='bold', pad=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🎯 Strongest Correlations with Risk")
                    target_corr = corr_matrix['Risk_Probability'].drop('Risk_Probability').abs().sort_values(ascending=False)
                    for i, (feature, corr_val) in enumerate(target_corr.head(5).items()):
                        correlation_strength = "Very Strong" if corr_val > 0.7 else "Strong" if corr_val > 0.5 else "Moderate" if corr_val > 0.3 else "Weak"
                        st.write(f"{i+1}. {feature}:** {corr_val:.3f} ({correlation_strength})")
                with col2:
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    target_corr.head(8).plot(kind='bar', ax=ax4, color='coral', alpha=0.8)
                    ax4.set_title('Top Feature Correlations with Risk', fontweight='bold', fontsize=14)
                    ax4.set_ylabel('Absolute Correlation', fontsize=12)
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.grid(True, axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close()
            else:
                st.warning("⚠ Cannot generate correlation matrix: need at least two numeric columns (Age, RestingBP, Cholesterol, MaxHR, Oldpeak, Risk_Probability).")

        
        elif selected_viz == "Predictive Insights":
            st.markdown("### 🔬 Predictive Insights & Model Confidence")
            
            # Since the actual feature importances are not stored in the session state, we use mock/placeholder values
            st.markdown("#### Feature Importance Analysis (Placeholder)")
            feature_names = ['Oldpeak', 'MaxHR', 'Age', 'Cholesterol', 'RestingBP', 'Sex', 'FastingBS', 
                             'ChestPain_ASY', 'ChestPain_NAP', 'ST_Slope_Flat', 'ExerciseAngina']
            # Mock importances reflecting expected pattern
            importance_scores = [0.15, 0.12, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02] 
            
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            bars = ax1.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', alpha=0.8)
            ax1.set_title('Feature Importance (Simulated)', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Importance Score', fontsize=12)
            ax1.set_ylabel('Feature', fontsize=12)
            for bar, score in zip(bars, importance_df['Importance']):
                ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{score:.2f}', ha='left', va='center', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()
            
            st.markdown("#### Prediction Confidence Distribution")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.histplot(df_viz['Risk_Probability'], bins=20, kde=True, ax=ax2, color='#667eea', alpha=0.8)
            ax2.set_title('Distribution of Model Prediction Confidence', fontweight='bold', fontsize=16)
            ax2.set_xlabel('Risk Probability', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Frequency', fontweight='bold', fontsize=12)
            ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
    else:
        st.info("📤 Please upload and process data in the 'Bulk Analysis' tab to see analytics here.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Heart Disease Prediction App v2.1</strong></p>
    <p>Professional AI-powered cardiovascular risk assessment tool</p>
    <p style='font-size: 12px; color: #999; margin-top: 10px;'>
        ⚠ This system is for educational and research purposes only. 
        Always consult healthcare professionals for medical decisions.
    </p>
</div>
""", unsafe_allow_html=True)
