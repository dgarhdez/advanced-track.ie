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
    
    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Fetch KPI Data
        rev_df = db.get_query_as_df(queries.GET_TOTAL_REVENUE)
        fleet_df = db.get_query_as_df(queries.GET_ACTIVE_FLEET_COUNT)
        load_df = db.get_query_as_df(queries.GET_AVG_LOAD_FACTOR)
        asm_df = db.get_query_as_df(queries.GET_TOTAL_ASM)
        
        total_revenue = rev_df['TOTAL_REVENUE'].iloc[0] if not rev_df.empty else 0
        active_fleet = fleet_df['FLEET_COUNT'].iloc[0] if not fleet_df.empty else 0
        avg_load_factor = load_df['AVG_LOAD_FACTOR'].iloc[0] if not load_df.empty else 0
        total_asm = asm_df['TOTAL_ASM'].iloc[0] if not asm_df.empty else 1 # Avoid div by zero
        
        rasm = total_revenue / total_asm if total_asm > 0 else 0
        
        with col1:
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col2:
            st.metric("Avg Load Factor", f"{avg_load_factor:.1f}%")
        with col3:
            st.metric("Active Fleet", f"{active_fleet}")
        with col4:
            st.metric("RASM", f"${rasm:.4f}", help="Revenue per Available Seat Mile")
            
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")

    st.markdown("---")
    
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
    
    try:
        route_df = db.get_query_as_df(queries.GET_ROUTE_NETWORK)
        
        if route_df.empty:
            st.warning("No route data found.")
        else:
            import plotly.graph_objects as go
            
            fig = go.Figure()

            # Add lines for each route
            for _, row in route_df.iterrows():
                fig.add_trace(
                    go.Scattergeo(
                        lat=[row['ORIGIN_LAT'], row['DEST_LAT']],
                        lon=[row['ORIGIN_LON'], row['DEST_LON']],
                        mode='lines',
                        line=dict(width=1, color='blue'),
                        opacity=0.4,
                        hoverinfo='text',
                        text=f"{row['ORIGIN']} ➡️ {row['DEST']}<br>Flights: {row['FLIGHT_COUNT']}"
                    )
                )

            # Add markers for airports
            # Combine origin and destination for unique markers
            airports = pd.concat([
                route_df[['ORIGIN', 'ORIGIN_LAT', 'ORIGIN_LON']].rename(columns={'ORIGIN': 'iata', 'ORIGIN_LAT': 'lat', 'ORIGIN_LON': 'lon'}),
                route_df[['DEST', 'DEST_LAT', 'DEST_LON']].rename(columns={'DEST': 'iata', 'DEST_LAT': 'lat', 'DEST_LON': 'lon'})
            ]).drop_duplicates()

            fig.add_trace(
                go.Scattergeo(
                    lat=airports['lat'],
                    lon=airports['lon'],
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    text=airports['iata'],
                    hoverinfo='text'
                )
            )

            fig.update_layout(
                title_text='Global Flight Network',
                showlegend=False,
                geo=dict(
                    scope='world',
                    projection_type='equirectangular',
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    countrycolor='rgb(204, 204, 204)'
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Show Raw Data
            with st.expander("Show Route Data"):
                st.dataframe(route_df)

    except Exception as e:
        st.error(f"Error visualizing routes: {e}")

elif selected_page == "Fleet Manager":
    st.write("### Fleet Status & Maintenance")
    
    # Fetch Airplane Data
    try:
        airplanes_df = db.get_query_as_df(queries.GET_AIRPLANES)
        
        if airplanes_df.empty:
            st.warning("No fleet data found.")
        else:
            # Maintenance Logic
            # Let's say thresh is 700 hours or 900 takeoffs
            airplanes_df['Maintenance Alert'] = (
                (airplanes_df['MAINTENANCE_FLIGHT_HOURS'] > 700) | 
                (airplanes_df['MAINTENANCE_TAKEOFFS'] > 900)
            )
            
            # Summary Metrics
            needs_maint = airplanes_df['Maintenance Alert'].sum()
            total_fleet = len(airplanes_df)
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Total Aircraft", total_fleet)
            m_col2.metric("Pending Maintenance", f"{needs_maint}", delta=-int(needs_maint), delta_color="inverse")
            
            st.markdown("---")
            st.subheader("Aircraft Status List")
            
            def highlight_maint(row):
                return ['background-color: #ffcccc' if row['Maintenance Alert'] else '' for _ in row]

            # Style the dataframe
            styled_fleet = airplanes_df.style.apply(highlight_maint, axis=1)
            st.dataframe(styled_fleet)
            
            if needs_maint > 0:
                st.warning(f"🚨 {needs_maint} aircraft require immediate maintenance attention!")

    except Exception as e:
        st.error(f"Error fetching fleet data: {e}")

elif selected_page == "Financials":
    st.write("### Financial Performance")
    
    try:
        # Fetch Financial Data
        rev_class_df = db.get_query_as_df(queries.GET_REVENUE_BY_CLASS)
        profit_df = db.get_query_as_df(queries.GET_ROUTE_PROFITABILITY)
        
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            st.subheader("Revenue by Class")
            if not rev_class_df.empty:
                import plotly.express as px
                fig_rev = px.pie(rev_class_df, values='REVENUE', names='CLASS', hole=0.4,
                                 title="Revenue Distribution by Ticket Class")
                st.plotly_chart(fig_rev, use_container_width=True)
            else:
                st.info("No revenue data available.")
                
        with f_col2:
            st.subheader("Top 5 Profitable Routes")
            if not profit_df.empty:
                top_routes = profit_df.head(5)
                fig_profit = px.bar(top_routes, x='ROUTE_CODE', y='ESTIMATED_PROFIT',
                                   title="Estimated Profit by Route ($)",
                                   labels={'ESTIMATED_PROFIT': 'Est. Profit ($)', 'ROUTE_CODE': 'Route'},
                                   color='ESTIMATED_PROFIT', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_profit, use_container_width=True)
            else:
                st.info("No profitability data available.")
                
        st.markdown("---")
        st.subheader("Route Economics Breakdown")
        if not profit_df.empty:
            st.dataframe(profit_df.style.format({
                'TOTAL_REVENUE': '${:,.2f}',
                'ESTIMATED_FUEL_COST': '${:,.2f}',
                'ESTIMATED_PROFIT': '${:,.2f}'
            }))
        else:
            st.info("No detailed economics data available.")

    except Exception as e:
        st.error(f"Error analyzing finances: {e}")
