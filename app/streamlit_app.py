import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Valdýr - House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Headers */
    h1 {
        background: linear-gradient(90.7deg, rgb(255, 253, 218) 1.9%, rgb(246, 186, 255) 39.5%, rgb(155, 226, 255) 75.6%, rgb(255, 253, 218) 100.2%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    /* Text */
    p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #111827;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90.7deg, rgb(246, 186, 255) 0%, rgb(155, 226, 255) 100%) !important;
    }

    .stNumberInput > div > div > input {
        background-color: #1f2937;
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
    }

    .stCheckbox > label > div[role="checkbox"] {
        background-color: #1f2937;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {
        background-color: #8b5cf6;
        border: 1px solid #8b5cf6;
    }

    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #1f2937;
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90.7deg, rgb(246, 186, 255) 1.9%, rgb(155, 226, 255) 100.2%) !important;
        color: #1e293b !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(246, 186, 255, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(246, 186, 255, 0.5) !important;
        filter: brightness(1.1);
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Result Card */
    .result-card {
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        animation: scaleIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90.7deg, transparent, rgb(246, 186, 255), rgb(155, 226, 255), transparent);
    }

    .result-card h3 {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
    }

    .result-card h1 {
        font-size: 4.5rem;
        margin: 0;
        text-shadow: 0 0 30px rgba(246, 186, 255, 0.2);
    }

    /* Animations */
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9) translateY(20px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }

    /* Dividers */
    hr {
        border-color: rgba(255,255,255,0.05);
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner="Loading House Price Prediction Model...")
def load_model():
    model_path = BASE_DIR / "house_price_model.pkl"
    columns_path = BASE_DIR / "model_columns.pkl"
    
    if not model_path.exists() or not columns_path.exists():
        raise FileNotFoundError("Model files not found. Please ensure house_price_model.pkl and model_columns.pkl are in the app/ directory.")
        
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns

try:
    model, model_columns = load_model()
except Exception as e:
    st.error(f"❌ Error setting up prediction model: {e}")
    st.info("💡 Make sure you have trained the model and placed the .pkl files inside the `/app` folder.")
    st.stop()

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align: center;'>🏡 Valdýr: House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get an instant estimate for your property value using Machine Learning.</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.markdown("## ⚙️ House Features")
st.sidebar.markdown("Adjust the parameters to see the predicted price.")

area = st.sidebar.slider("📐 Area (sq ft)", 500, 10000, 2000, step=100)
bedrooms = st.sidebar.number_input("🛏️ Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.number_input("🛁 Bathrooms", 1, 5, 2)
stories = st.sidebar.number_input("🏢 Stories", 1, 4, 2)
parking = st.sidebar.number_input("🚗 Parking (Capacity)", 0, 3, 1)

st.sidebar.markdown("### 🌟 Amenities")
mainroad = st.sidebar.checkbox("🛣️ Near Main Road", value=True)
guestroom = st.sidebar.checkbox("🛌 Guest Room", value=False)
basement = st.sidebar.checkbox("🏚️ Basement", value=False)
hotwaterheating = st.sidebar.checkbox("🔥 Hot Water Heating", value=False)
airconditioning = st.sidebar.checkbox("❄️ Air Conditioning", value=True)
prefarea = st.sidebar.checkbox("📍 Preferred Area", value=True)

st.sidebar.markdown("### 🛋️ Furnishing Status")
furnishingstatus = st.sidebar.selectbox(
    "Select Furnishing",
    ["furnished", "semi-furnished", "unfurnished"],
    label_visibility="collapsed"
)

# ---------------- DATAFRAME ---------------- #
input_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "mainroad": 1 if mainroad else 0,
    "guestroom": 1 if guestroom else 0,
    "basement": 1 if basement else 0,
    "hotwaterheating": 1 if hotwaterheating else 0,
    "airconditioning": 1 if airconditioning else 0,
    "prefarea": 1 if prefarea else 0,
    "furnishingstatus_semi-furnished": 1 if furnishingstatus == "semi-furnished" else 0,
    "furnishingstatus_unfurnished": 1 if furnishingstatus == "unfurnished" else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)

# Essential: align columns with identically to what the model expects
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ---------------- MAIN CONTENT ---------------- #
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Your Inputs")
    st.dataframe(
        pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}),
        width='stretch',
        height=400
    )

with col2:
    st.markdown("### 💰 Valuation")
    st.write("Click the button below to predict the current market price of the specified house.")
    
    if st.button("Predict Real Estate Price 🚀", use_container_width=True, type="primary"):
        with st.spinner("Analyzing market parameters..."):
            try:
                prediction = model.predict(input_df)[0]
                st.success("Analysis Complete!")
                st.markdown(f"""
                <div class="result-card">
                    <h3>Estimated Price</h3>
                    <h1>₹ {int(prediction):,}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
                
# ---------------- FOOTER ---------------- #
st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with ❤️ for rapid real estate prediction | Streamlit & scikit-learn</p>",
    unsafe_allow_html=True
)