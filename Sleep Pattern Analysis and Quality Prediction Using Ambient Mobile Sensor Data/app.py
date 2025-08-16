import streamlit as st
import pandas as pd
import joblib
import sklearn
import xgboost
import json
import time
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# ---------------- Security Configuration ----------------
def get_api_key():
    """Safely retrieve API key from environment variables"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ No Google Gemini API key found. AI recommendations will use rule-based fallback.")
        return None
    return api_key

# Set API key only if it exists in environment
api_key = get_api_key()
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="AI-Powered Sleep Quality Predictor",
    page_icon="🛌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    color: #333; /* Dark text for readability on white card */
}
.ai-suggestion {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.expert-tip {
    background: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    color: #000000; /* Dark text for readability */
}
.priority-high {
    border-left-color: #f44336;
}
.priority-medium {
    border-left-color: #ff9800;
}
.priority-low {
    border-left-color: #2196f3;
}
</style>
""", unsafe_allow_html=True)

# ---------------- AI Integration Setup (UPDATED FOR GEMINI) ----------------
class AIExpertSystem:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
            # Using Gemini 1.5 Flash, the fastest and most advanced model
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def generate_personalized_suggestions(self, user_data: Dict, prediction_confidence: float, expert_data: Optional[Dict] = None) -> List[Dict]:
        """Generate AI-powered personalized sleep suggestions"""
        sleep_efficiency = self._calculate_sleep_efficiency(user_data)
        risk_factors = self._identify_risk_factors(user_data)
        context = self._build_context(user_data, sleep_efficiency, risk_factors, prediction_confidence, expert_data)

        if self.model:
            return self._get_ai_suggestions(context)
        else:
            st.warning("Google Gemini API key not found. Using rule-based suggestions as a fallback.")
            return self._get_rule_based_suggestions(user_data, risk_factors)

    def _calculate_sleep_efficiency(self, user_data: Dict) -> float:
        total_time_in_bed = user_data['duration_in_bed_minutes']
        awake_time = user_data['in_bed_awake_duration_minutes'] + user_data['sleep_onset_latency_minutes']
        if total_time_in_bed <= 0: return 0.0
        sleep_time = max(0, total_time_in_bed - awake_time)
        return min(100, max(0, (sleep_time / total_time_in_bed) * 100))

    def _identify_risk_factors(self, user_data: Dict) -> List[str]:
        risks = []
        if user_data['sleep_onset_latency_minutes'] > 30: risks.append("prolonged_sleep_onset")
        if user_data['in_bed_awake_duration_minutes'] > 45: risks.append("sleep_fragmentation")
        if user_data['night_time_phone_usage_day_minutes'] > 60: risks.append("excessive_screen_time")
        if user_data['phone_unlock_count_day'] > 10: risks.append("frequent_awakenings")
        if user_data['duration_in_bed_minutes'] < 420: risks.append("insufficient_sleep_opportunity")
        return risks

    def _build_context(self, user_data: Dict, efficiency: float, risks: List[str], confidence: float, expert_data: Optional[Dict] = None) -> str:
        risk_str = ', '.join(risks) if risks else 'None identified'
        context_str = f"""
        You are an AI sleep expert. Your task is to provide personalized, actionable, and scientifically-backed sleep improvement suggestions based on user data.
        Analyze the following sleep data and provide 3-5 personalized, evidence-based sleep improvement suggestions.
        The output must be a valid JSON array of objects, with each object having keys: "title", "priority", "explanation", "action", and "science".
        The "priority" should be "high", "medium", or "low". Do not include any text before or after the JSON.

        Core Sleep Metrics:
        - Time in bed: {user_data['duration_in_bed_minutes']} minutes
        - Time to fall asleep: {user_data['sleep_onset_latency_minutes']} minutes
        - Awake time during night: {user_data['in_bed_awake_duration_minutes']} minutes
        - Night phone usage: {user_data['night_time_phone_usage_day_minutes']} minutes
        - Phone unlocks: {user_data['phone_unlock_count_day']} times
        - Calculated Sleep Efficiency: {efficiency:.1f}%
        - Identified Risk Factors: {risk_str}
        - Model Prediction Confidence (for poor sleep): {confidence:.1%}
        """

        if expert_data:
            context_str += "\n\nClinical and Lifestyle Factors (Expert Mode):\n"
            context_str += f"- Stress Level (1-10): {expert_data['stress_level']}\n"
            context_str += f"- Sleep Medication: {expert_data['sleep_medication']}\n"
            context_str += f"- Last Caffeine Intake: {expert_data['caffeine_timing']}\n"
            context_str += f"- Exercise Timing: {expert_data['exercise_timing']}\n"
            context_str += "Incorporate these clinical factors for more targeted and relevant advice."

        return context_str

    def _get_ai_suggestions(self, context: str) -> List[Dict]:
        """Get AI-generated suggestions using Gemini API"""
        try:
            # Set the generation config to enforce JSON output
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(context, generation_config=generation_config)

            suggestions_data = json.loads(response.text)

            # Handle cases where the JSON is nested inside a key (e.g., {"suggestions": [...]})
            if isinstance(suggestions_data, dict):
                for key in suggestions_data:
                    if isinstance(suggestions_data[key], list): return suggestions_data[key]

            if isinstance(suggestions_data, list): return suggestions_data

            return self._get_rule_based_suggestions({}, [])
        except Exception as e:
            st.error(f"❌ Error getting AI suggestions from Gemini: {e}")
            return self._get_rule_based_suggestions({}, [])

    def _get_rule_based_suggestions(self, user_data: Dict, risks: List[str]) -> List[Dict]:
        return [{"title": "Maintain Consistent Sleep Schedule", "priority": "high", "explanation": "Regular sleep timing strengthens your circadian rhythm.", "action": "Go to bed and wake up at the same time every day, even on weekends.", "science": "Consistent timing helps regulate your internal biological clock."}]

# --- NEW FUNCTION: To generate the text for the download button ---
def generate_report_string(user_input, expert_data, prediction_label, poor_sleep_prob, sleep_efficiency, suggestions):
    """Creates a formatted string for the downloadable report."""
    report = []
    report.append("="*40)
    report.append("        AI-POWERED SLEEP ANALYSIS REPORT")
    report.append("="*40)
    report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("-" * 20 + " SUMMARY " + "-" * 20)
    report.append(f"Overall Prediction: {prediction_label}")
    report.append(f"Risk of Poor Sleep: {poor_sleep_prob:.1%}")
    report.append(f"Calculated Sleep Efficiency: {sleep_efficiency:.1f}%\n")

    report.append("-" * 20 + " YOUR INPUTS " + "-" * 20)
    report.append(f"- Total Time in Bed: {user_input['duration_in_bed_minutes']} min")
    report.append(f"- Time to Fall Asleep: {user_input['sleep_onset_latency_minutes']} min")
    report.append(f"- Time Awake During Night: {user_input['in_bed_awake_duration_minutes']} min")
    report.append(f"- Phone Usage in Bed: {user_input['night_time_phone_usage_day_minutes']} min")
    report.append(f"- Phone Unlocks: {user_input['phone_unlock_count_day']}\n")

    if expert_data:
        report.append("-" * 20 + " EXPERT MODE INPUTS " + "-" * 20)
        report.append(f"- Stress Level (1-10): {expert_data['stress_level']}")
        report.append(f"- Sleep Medication: {expert_data['sleep_medication']}")
        report.append(f"- Last Caffeine: {expert_data['caffeine_timing']}")
        report.append(f"- Exercise Timing: {expert_data['exercise_timing']}\n")

    report.append("=" * 40)
    report.append("       PERSONALIZED AI RECOMMENDATIONS")
    report.append("=" * 40)
    if suggestions:
        for i, sug in enumerate(suggestions, 1):
            report.append(f"\n--- Suggestion #{i} (Priority: {sug.get('priority', 'N/A').upper()}) ---")
            report.append(f"Title: {sug.get('title', 'N/A')}")
            report.append(f"Explanation: {sug.get('explanation', 'N/A')}")
            report.append(f"Action: {sug.get('action', 'N/A')}")
            report.append(f"The Science: {sug.get('science', 'N/A')}")
    else:
        report.append("No specific suggestions were generated for this analysis.")

    return "\n".join(report)

# ---------------- Input Validation Functions ----------------
def validate_sleep_inputs(duration_in_bed: int, sleep_onset_latency: int, 
                         in_bed_awake_duration: int, night_time_phone_usage: int, 
                         phone_unlock_count: int) -> tuple[bool, str]:
    """Validate user inputs and return (is_valid, error_message)"""
    
    # Check for logical consistency
    total_awake_time = sleep_onset_latency + in_bed_awake_duration
    if total_awake_time > duration_in_bed:
        return False, "❌ Total awake time cannot exceed total time in bed"
    
    if sleep_onset_latency > duration_in_bed:
        return False, "❌ Time to fall asleep cannot exceed total time in bed"
    
    if night_time_phone_usage > duration_in_bed:
        return False, "❌ Phone usage time cannot exceed total time in bed"
    
    # Check for reasonable ranges
    if duration_in_bed < 180:  # Less than 3 hours
        return False, "❌ Time in bed seems too short (minimum 3 hours recommended)"
    
    if duration_in_bed > 720:  # More than 12 hours
        return False, "❌ Time in bed seems too long (maximum 12 hours recommended)"
    
    if sleep_onset_latency > 120:  # More than 2 hours
        return False, "❌ Time to fall asleep seems unusually long"
    
    if phone_unlock_count > 50:
        return False, "❌ Phone unlock count seems unusually high"
    
    return True, "✅ Inputs validated successfully"

def safe_model_prediction(model, input_df: pd.DataFrame, model_features: list) -> tuple[bool, any, any, str]:
    """Safely make model predictions with comprehensive error handling"""
    try:
        # Validate input dataframe
        if input_df.empty:
            return False, None, None, "❌ No input data provided"
        
        # Check if all required features are present
        missing_features = [f for f in model_features if f not in input_df.columns]
        if missing_features:
            return False, None, None, f"❌ Missing features: {', '.join(missing_features)}"
        
        # Check for NaN values
        if input_df[model_features].isnull().any().any():
            return False, None, None, "❌ Input data contains missing values"
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        return True, prediction, probabilities, "✅ Prediction successful"
        
    except AttributeError:
        return False, None, None, "❌ Model not properly loaded"
    except ValueError as e:
        return False, None, None, f"❌ Invalid input data: {str(e)}"
    except Exception as e:
        return False, None, None, f"❌ Prediction error: {str(e)}"

# ---------------- Model Loading ----------------
@st.cache_resource
def load_model_and_ai():
    model = None
    ai_system = None
    try:
        with open('sleep_quality_binary_model.pkl', 'rb') as f: model = joblib.load(f)
        api_key = os.environ.get("GOOGLE_API_KEY")
        ai_system = AIExpertSystem(api_key=api_key)
    except FileNotFoundError:
        st.error("❌ Model file not found. Prediction is disabled.")
        ai_system = AIExpertSystem(api_key=os.environ.get("GOOGLE_API_KEY"))
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        ai_system = AIExpertSystem(api_key=os.environ.get("GOOGLE_API_KEY"))
    return model, ai_system

model, ai_expert = load_model_and_ai()

# ---------------- UI Layout ----------------
st.markdown("<div class='main-header'><h1>🛌 AI-Powered Sleep Quality Predictor</h1><p>Get personalized, expert-level sleep recommendations powered by artificial intelligence.</p></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])

# --- Column 1: Inputs ---
with col1:
    st.header("📊 Input Your Sleep Metrics")
    st.subheader("⏰ Sleep Timing")
    duration_in_bed = st.slider("Total Time in Bed (minutes)", 180, 720, 480, 15, help="From getting into bed to getting out of bed")
    sleep_onset_latency = st.slider("Time to Fall Asleep (minutes)", 0, 120, 20, 5, help="How long after getting in bed did you fall asleep?")
    st.subheader("🌙 Sleep Quality")
    in_bed_awake_duration = st.slider("Time Awake During Night (minutes)", 0, 180, 30, 5, help="Total time awake during the night")
    st.subheader("📱 Technology Usage")
    night_time_phone_usage = st.slider("Phone Usage in Bed (minutes)", 0, 180, 30, 5, help="Active phone use while in bed")
    phone_unlock_count = st.number_input("Phone Unlocks During Night", 0, 50, 3, 1, help="Number of times you checked your phone")

# --- Column 2: Live Metrics ---
with col2:
    st.header("📈 Live Metrics")
    user_input = {'duration_in_bed_minutes': duration_in_bed, 'sleep_onset_latency_minutes': sleep_onset_latency, 'in_bed_awake_duration_minutes': in_bed_awake_duration, 'night_time_phone_usage_day_minutes': night_time_phone_usage, 'phone_unlock_count_day': phone_unlock_count}
    total_awake_time = sleep_onset_latency + in_bed_awake_duration
    estimated_sleep_time = max(0, duration_in_bed - total_awake_time)
    sleep_efficiency = (estimated_sleep_time / duration_in_bed * 100) if duration_in_bed > 0 else 0
    st.metric("Sleep Efficiency", f"{sleep_efficiency:.1f}%", help="Percentage of time in bed actually sleeping")
    st.metric("Estimated Sleep Time", f"{estimated_sleep_time:.0f} min", help="Approximate total sleep duration")
    screen_impact = "High" if night_time_phone_usage > 60 else "Medium" if night_time_phone_usage > 30 else "Low"
    st.metric("Screen Time Impact", screen_impact)
    onset_category = "Fast" if sleep_onset_latency <= 15 else "Normal" if sleep_onset_latency <= 30 else "Slow"
    st.metric("Sleep Onset", onset_category)

# --- Column 3: Prediction ---
with col3:
    st.header("🎯 Prediction & Analysis")
    model_features = ['duration_in_bed_minutes', 'sleep_onset_latency_minutes', 'in_bed_awake_duration_minutes', 'night_time_phone_usage_day_minutes', 'phone_unlock_count_day']
    if st.button("🔮 Analyze Sleep Quality", type="primary", use_container_width=True, disabled=(model is None)):
        input_df = pd.DataFrame([user_input])[model_features]
        is_valid, validation_message = validate_sleep_inputs(duration_in_bed, sleep_onset_latency, in_bed_awake_duration, night_time_phone_usage, phone_unlock_count)
        if not is_valid:
            st.error(validation_message)
        else:
            success, prediction, probabilities, prediction_message = safe_model_prediction(model, input_df, model_features)
            if success:
                poor_sleep_prob = probabilities[1] # probabilities[1] is the probability of class 1 (poor sleep)
                prediction_label = "Not Good Sleep" if prediction == 1 else "Good Sleep"
                st.session_state['last_prediction'] = {"prediction_label": prediction_label, "poor_sleep_prob": poor_sleep_prob, "sleep_efficiency": sleep_efficiency, "suggestions": None}
                #st.success(f"✅ {prediction_message}")
            else:
                st.error(f"❌ {prediction_message}")

    if 'last_prediction' in st.session_state:
        pred_info = st.session_state['last_prediction']
        if pred_info['prediction_label'] == "Good Sleep":
            st.success("✅ **GOOD SLEEP PREDICTED**")
            st.write(f"**Confidence:** {1 - pred_info['poor_sleep_prob']:.1%}")
        else:
            st.warning("⚠️ **POOR SLEEP PREDICTED**")
            st.write(f"**Risk Probability:** {pred_info['poor_sleep_prob']:.1%}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred_info['sleep_efficiency'],
            title={'text': "Sleep Efficiency Gauge"}, delta={'reference': 85},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#667eea"}, 'steps': [{'range': [0, 70], 'color': "#f5576c"}, {'range': [70, 85], 'color': "orange"}, {'range': [85, 100], 'color': "green"}]}
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --- Sidebar Logic (must be defined before being used in the AI button) ---
expert_data = None
st.sidebar.markdown("## 🔧 Advanced Settings")
if st.sidebar.checkbox("Enable Expert Mode"):
    st.sidebar.markdown("### 📋 Clinical Sleep Assessment")
    expert_data = {
        'sleep_medication': st.sidebar.selectbox("Sleep Medication Use", ["None", "Occasional", "Regular", "Prescribed"]),
        'stress_level': st.sidebar.slider("Stress Level (1-10)", 1, 10, 5),
        'caffeine_timing': st.sidebar.selectbox("Last Caffeine Intake", ["Morning only", "Early afternoon", "Late afternoon", "Evening"]),
        'exercise_timing': st.sidebar.selectbox("Exercise Timing", ["Morning", "Afternoon", "Early evening", "Late evening", "None"])
    }

# --- AI Suggestions Section ---
st.header("🧠 AI-Powered Sleep Coaching")
if st.button("🤖 Get Personalized AI Suggestions", use_container_width=True):
    if 'last_prediction' in st.session_state:
        with st.spinner("🚀 Gemini AI is analyzing your sleep patterns..."):
            poor_sleep_prob_for_ai = st.session_state['last_prediction']['poor_sleep_prob']
            suggestions = ai_expert.generate_personalized_suggestions(user_input, poor_sleep_prob_for_ai, expert_data)
            st.session_state['last_prediction']['suggestions'] = suggestions
            st.success("✨ AI Analysis Complete!")
            for i, suggestion in enumerate(suggestions, 1):
                priority_class = f"priority-{suggestion.get('priority', 'medium')}"
                with st.container():
                    st.markdown(f"""<div class="metric-card {priority_class}"><h4>{suggestion.get('title', 'Suggestion')}</h4><p><strong>Why this matters:</strong> {suggestion.get('explanation', 'N/A')}</p><p><strong>What to do:</strong> {suggestion.get('action', 'N/A')}</p><p><em>💡 Science: {suggestion.get('science', 'N/A')}</em></p></div>""", unsafe_allow_html=True)
    else:
        st.warning("Please 'Analyze Sleep Quality' first to generate a prediction.")

# --- The rest of your UI (Charts, Tips, Footer) ---
st.header("📊 Sleep Pattern Analysis")
col1_trends, col2_trends = st.columns(2)
with col1_trends:
    dates = pd.date_range(start=datetime.now() - timedelta(days=6), end=datetime.now(), freq='D')
    mock_efficiency = [sleep_efficiency + np.random.normal(0, 5) for _ in range(7)]
    mock_efficiency = [max(0, min(100, eff)) for eff in mock_efficiency]
    fig_trend = px.line(x=dates, y=mock_efficiency, title="Sleep Efficiency Trend (7 Days)", labels={'x': 'Date', 'y': 'Sleep Efficiency (%)'})
    fig_trend.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Optimal (85%+) ")
    fig_trend.update_layout(height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

with col2_trends:
    sleep_components = {'Estimated Sleep': estimated_sleep_time, 'Time to Fall Asleep': sleep_onset_latency, 'Night Awakenings': in_bed_awake_duration, 'Phone Usage': night_time_phone_usage}
    fig_pie = px.pie(values=list(sleep_components.values()), names=list(sleep_components.keys()), title="Time Distribution in Bed")
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

st.header("💡 Sleep Medicine Insights")
expert_tips = [{"title": "The 3-2-1 Rule", "tip": "No food 3 hours before bed, no liquids 2 hours before, no screens 1 hour before.", "science": "This prevents digestive disruption, reduces nighttime bathroom trips, and minimizes blue light exposure."}, {"title": "Temperature Optimization", "tip": "Keep your bedroom between 65-68°F (18-20°C) for optimal sleep.", "science": "Core body temperature naturally drops 1-2°F during sleep onset. A cool environment facilitates this process."}, {"title": "Light Exposure Timing", "tip": "Get bright light within 30 minutes of waking, dim lights 2 hours before bed.", "science": "Light is the strongest zeitgeber (time cue) for your circadian rhythm, affecting melatonin production."}]
for tip in expert_tips:
    st.markdown(f"""<div class="expert-tip"><h4>🏥 {tip['title']}</h4><p><strong>Recommendation:</strong> {tip['tip']}</p><p><em>Why it works:</em> {tip['science']}</em></p></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div style='text-align: center; color: #666; padding: 2rem;'><p>🛌 <strong>AI-Powered Sleep Quality Predictor</strong></p><p>Combining machine learning predictions with expert sleep medicine knowledge</p><p><em>For educational purposes. Consult a sleep specialist for persistent sleep issues.</em></p></div>""", unsafe_allow_html=True)

# --- Final Sidebar Section ---
if st.sidebar.checkbox("Configure AI Settings"):
    st.sidebar.markdown("### 🤖 AI Configuration")
    api_key_input = st.sidebar.text_input("Enter Your Gemini API Key", type="password", help="Enter your Gemini API key if you want to use your own.")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.sidebar.success("✅ Custom Gemini API Key is set!")
    else:
        st.sidebar.info("ℹ️ Using the built-in application API key.")

if 'last_prediction' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📄 Your Report")

    report_data_string = generate_report_string(
        user_input, expert_data,
        st.session_state['last_prediction']['prediction_label'],
        st.session_state['last_prediction']['poor_sleep_prob'],
        st.session_state['last_prediction']['sleep_efficiency'],
        st.session_state['last_prediction'].get('suggestions')
    )

    st.sidebar.download_button(
        label="📥 Export Sleep Report",
        data=report_data_string,
        file_name=f"sleep_report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )


