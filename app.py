import streamlit as st
import pandas as pd
import queries
from database import DatabaseConnector

# Initialize Database Connection
# Using st.cache_resource to initialize the database connection only once
@st.cache_resource
def get_database_connection():
    return DatabaseConnector()

db = get_database_connection()

# Set Page Config
st.set_page_config(
    page_title="SkyHigh Insights",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
with st.sidebar:
    st.title("👨‍✈️ SkyHigh Insights")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        ["Executive Summary", "Route Network", "Fleet Manager", "Financials"]
    )
    
    st.markdown("---")
    st.caption("v1.0.0 | Connected to IBM DB2")

# Main Page Content
st.title(f"{selected_page}")

if selected_page == "Executive Summary":
    st.write("### Welcome to the Executive Command Center")
    st.write("Here you will see high-level metrics and KPIs.")
    
    # Example Data Fetch
    st.subheader("Recent Flights (Live Data)")
    try:
        # Use the helper method which handles DataFrame creation
        flights_df = db.get_query_as_df(queries.GET_FLIGHTS)
        
        if flights_df.empty:
            st.warning("No flight data found.")
        else:
            st.dataframe(flights_df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")

elif selected_page == "Route Network":
    st.write("### Route Network Visualization")
    st.info("Interactive map coming soon...")

elif selected_page == "Fleet Manager":
    st.write("### Fleet Status & Maintenance")
    
    # Fetch Airplane Data
    try:
        airplanes_df = db.get_query_as_df(queries.GET_AIRPLANES)
        st.dataframe(airplanes_df)
    except Exception as e:
        st.error(f"Error fetching fleet data: {e}")

elif selected_page == "Financials":
    st.write("### Financial Performance")
    st.info("Revenue analysis coming soon...")
