import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# 2. Load Models
@st.cache_resource
def load_assets():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return kmeans, scaler

kmeans, scaler = load_assets()

# 3. Custom CSS for Styling
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    h1 {color: #9b59b6; padding-bottom: 1rem;}
    .result-card {
        background-color: #9b59b6; 
        padding: 2rem; 
        border-radius: 1rem; 
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 4. Sidebar for Inputs
st.sidebar.title("Customer Profile")
st.sidebar.subheader("Enter Customer Details")

age = st.sidebar.number_input("Age", 18, 100, 35)
income = st.sidebar.number_input("Income", 0, 200000, 50000)
total_spending = st.sidebar.number_input("Total Spending", 0, 5000, 100)
num_web_purchases = st.sidebar.number_input("Web Purchases", 0, 100, 10)
num_store_purchases = st.sidebar.number_input("Store Purchases", 0, 100, 10)
num_web_visits = st.sidebar.number_input("Web Visits/Month", 0, 50, 3)
recency = st.sidebar.number_input("Recency (Days)", 0, 365, 30)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Find Customer Segment", type="primary", use_container_width=True)

# 5. Main Content Area
st.title("Customer Segmentation System")
st.markdown("### AI-Powered Marketing Intelligence")

if predict_btn:
    # Prepare and Scale Data
    input_data = pd.DataFrame({
        "Age": [age], "Income": [income], "Total_Spending": [total_spending],
        "NumWebPurchases": [num_web_purchases], "NumStorePurchases": [num_store_purchases],
        "NumWebVisitsMonth": [num_web_visits], "Recency": [recency]
    })
    
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    # Display Results
    st.markdown("---")
    st.header("Segmentation Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Result Card (Style matches PDF Page 4)
        st.markdown(f"""
            <div class="result-card">
                <h2 style='color: white; margin: 0;'>Predicted Segment: Cluster {cluster}</h2>
                <h4 style='color: white; margin: 10px 0;'>AI-Generated Customer Group</h4>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Marketing Strategy")
        st.info(f"Recommended action for Cluster {cluster}: Focus on personalized engagement based on their spending patterns.")

    with col2:
        st.markdown("### Model Insights")
        st.metric("Algorithm", "K-Means")
        st.metric("Input Features", "7")

else:
    # Initial landing view (Matches PDF Page 8)
    st.markdown("---")
    st.info("Enter customer details in the sidebar to find their segment.")
    
    # Placeholder for Segment Overview
    st.subheader("Segment Comparison")
    st.write("Predicted clusters will appear here after calculation.")
