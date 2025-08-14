import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.inference import predict_segment
from src.labeler import assign_segment_labels
from src.utils_io import load_config

st.set_page_config(layout="wide")
CONFIG = load_config()

@st.cache_data
def load_data():
    df = pd.read_csv(CONFIG['data']['processed_path'])
    return df

st.title("📈 Advanced CLV & Customer Segmentation Dashboard")

df = load_data()

# --- Assign segments to all data for visualization ---
segments = predict_segment(df)
df['segment'] = segments
df, label_map = assign_segment_labels(df)

# --- Main Dashboard ---
col1, col2 = st.columns(2)

with col1:
    st.header("Segment Distribution")
    segment_counts = df['segment_label'].value_counts()
    fig_pie = px.pie(
        values=segment_counts.values, 
        names=segment_counts.index, 
        title="Customer Segments"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.header("CLV Distribution by Segment")
    fig_box = px.box(
        df, 
        x='segment_label', 
        y='CLV_90_days', 
        color='segment_label',
        title="90-Day CLV by Segment"
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.header("Customer Lookup")
customer_id_input = st.selectbox("Select a CustomerID to inspect:", df['CustomerID'].unique())

if customer_id_input:
    customer_data = df[df['CustomerID'] == customer_id_input].iloc[0]
    
    st.subheader(f"Profile for Customer: {customer_id_input}")
    
    # Display key metrics
    st.metric("Assigned Segment", customer_data['segment_label'])
    st.metric("Predicted 90-Day CLV", f"${customer_data['CLV_90_days']:.2f}")
    st.metric("Probabilistic 90-Day CLV", f"${customer_data['probabilistic_clv_90d']:.2f}")

    with st.expander("View all features for this customer"):
        st.dataframe(customer_data)