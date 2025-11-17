
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="PH Flood Control Project Tracker", page_icon="🚨", layout="wide")

# --- CSS styling ---
st.markdown("""
<style>
html, body, [class*="css"] { 
    font-family: 'Inter', system-ui, sans-serif; 
}

section[data-testid="stSidebar"] { 
    background-color: #111217; 
    padding: 18px; 
}

section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { 
    color: #ffffff; 
}

.metric-label { 
    color: #9aa0a6; 
    font-size: 13px; 
    margin-bottom: 6px; 
}

.metric-value { 
    font-size: 28px; 
    font-weight: 700; 
    color: #ffffff; 
}

.muted { 
    color: #9aa0a6; 
    font-size: 13px; 
}

.top-menu { 
    position: absolute; 
    right: 20px; 
    top: 15px; 
}

.top-menu button { 
    color: #ffffff; 
    margin-left: 18px; 
    font-weight: 600; 
    background: none; 
    border: none; 
    cursor: pointer; 
    font-size: 16px; 
}

.top-menu button:hover { 
    text-decoration: underline; 
}

div[data-baseweb="tab-list"] { 
    background-color: #0f1113; 
    padding: 6px; border-radius: 
    10px; margin-bottom: 12px; 
}

[data-testid="stDataFrame"] { 
    border-radius: 10px; 
    overflow: hidden; 
}

.stSelectbox > div, .stMultiSelect > div, .stSlider > div { 
    margin-bottom: 8px; 
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-menu">
</div>
""", unsafe_allow_html=True)

# --- Page state for top-right tabs ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Top-right buttons ---
cols = st.columns([8, 1, 1])  # spacer, Home button, About button
with cols[1]:
    if st.button("Home"):
        st.session_state.page = "Home"
with cols[2]:
    if st.button("About"):
        st.session_state.page = "About"

# --- Card helpers ---
def card_begin(): st.markdown("<div class='card'>", unsafe_allow_html=True)
def card_end(): st.markdown("</div>", unsafe_allow_html=True)
def metric_card(label, value, help_text=None):
    st.markdown(f"<div class='metric-label'>{label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
    if help_text:
        st.markdown(f"<div class='muted'>{help_text}</div>", unsafe_allow_html=True)

# --- Load data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("flood_control_projects.csv")
        # Convert dates
        for col in ["CompletionDateActual", "StartDate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # Derived metrics
        if "CompletionDateActual" in df.columns and "StartDate" in df.columns:
            df["Duration_Days"] = (df["CompletionDateActual"] - df["StartDate"]).dt.days
        if "ABC" in df.columns and "ContractCost" in df.columns:
            df["Cost_Variance"] = df["ABC"] - df["ContractCost"]
            df["Cost_Variance_Pct"] = np.where(df["ABC"].replace(0, np.nan).notna(), (df["Cost_Variance"]/df["ABC"])*100, np.nan)
        if "ContractCost" in df.columns and "Duration_Days" in df.columns:
            df["Cost_Per_Day"] = df["ContractCost"]/df["Duration_Days"].replace(0, np.nan)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# --- Sidebar ---
st.sidebar.title("PH Flood Control Projects")
st.sidebar.markdown("**Filters & Alerts**")

with st.sidebar.expander("Filters", expanded=True):
    
    selected_region = None
    if "Region" in df.columns:
        regions = sorted(df["Region"].dropna().unique().tolist())
        selected_region = st.selectbox(
            "Select Region", 
            options=["All"]+regions, 
            index=0
        )

        if selected_region == "All":
            selected_region = None 

    # --- Province filter, dynamically based on selected region ---
    selected_province = None
    if "Province" in df.columns:
        if selected_region:
            provinces = sorted(df[df["Region"]==selected_region]["Province"].dropna().unique().tolist())
        else:
            provinces = sorted(df["Province"].dropna().unique().tolist())
        provinces_options = ["All"] + provinces
        selected_province = st.selectbox(
            "Select Province", 
            options=provinces_options, 
            index=0
        )
        if selected_province == "All":
            selected_province = None

    # --- Year filter ---
    selected_years = None
    if "InfraYear" in df.columns:
        if selected_region and selected_province:
            years = sorted(
                df[
                    (df["Region"] == selected_region) & (df["Province"] == selected_province)
                ]["InfraYear"].dropna().unique().tolist()
            )
        elif selected_region:
            years = sorted(df[df["Region"] == selected_region]["InfraYear"].dropna().unique().tolist())
        elif selected_province:
            years = sorted(df[df["Province"] == selected_province]["InfraYear"].dropna().unique().tolist())
        else:
            years = sorted(df["InfraYear"].dropna().unique().tolist())
        selected_years = st.multiselect(
            "Select Year(s)", 
            options=years, 
            default=[]
        )


# Alert thresholds
with st.sidebar.expander("Alert Thresholds", expanded=False):
    suspicious_savings = st.slider("Suspicious Cost Savings (%)", 0.0, 30.0, 8.0, step=1.0)
    duration_threshold = st.slider("Long Duration Alert (days)", 100, 1000, 365, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("Made for transparency • Data: bettergov.ph")

# --- Apply filters ---
filtered_df = df.copy()

# Apply Region filter
if selected_region and selected_region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]

# Apply Province filter
if selected_province and selected_province != "All":
    filtered_df = filtered_df[filtered_df["Province"] == selected_province]

# Apply Years filter
if selected_years:
    filtered_df = filtered_df[filtered_df["InfraYear"].isin(selected_years)]


# --- Anomaly detection ---
@st.cache_data
def detect_anomalies(data):
    features = ["ABC", "ContractCost", "Duration_Days", "Cost_Variance_Pct"]
    available_features = [f for f in features if f in data.columns and data[f].notna().any()]
    if len(available_features)<2 or len(data.dropna(subset=available_features))<10:
        data["Anomaly"] = 0; data["Anomaly_Score"] = 0; return data, pd.DataFrame()
    X = data.dropna(subset=available_features)[available_features]
    X_scaled = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination=0.1, random_state=42)
    labels = iso.fit_predict(X_scaled)
    scores = iso.score_samples(X_scaled)
    data.loc[X.index, ["Anomaly","Anomaly_Score"]] = list(zip(labels,scores))
    data["Anomaly"].fillna(0, inplace=True)
    data["Anomaly_Score"].fillna(0, inplace=True)
    anomalies = data[data["Anomaly"]==-1].copy()
    return data, anomalies

analysis_df, anomalies = detect_anomalies(filtered_df)

if analysis_df.shape[0]==0:
    st.warning("No data matches selected filters. Adjust filters in the sidebar.")
    st.stop()

# --- Main header ---
if st.session_state.page == "Home":
    st.markdown("# Philippines Flood Control Projects Tracker")
    st.markdown("### Transparency Dashboard • Red flags, trends, and contractor insights")

    # Red flag metrics in a single row card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    with cols[0]:
        high_cost_projects = int((analysis_df["ABC"] > analysis_df["ABC"].quantile(0.9)).sum()) if "ABC" in analysis_df.columns else 0
        metric_card("High-Value Projects (Top 10%)", f"{high_cost_projects:,}", "Large public expenditures")
    with cols[1]:
        suspicious_projects = int((analysis_df["Cost_Variance_Pct"] > suspicious_savings).sum()) if "Cost_Variance_Pct" in analysis_df.columns else 0
        metric_card("High Cost Savings", f"{suspicious_projects:,}", "Projects exceeding savings threshold")
    with cols[2]:
        delayed_projects = int((analysis_df["Duration_Days"] > duration_threshold).sum()) if "Duration_Days" in analysis_df.columns else 0
        metric_card("Delayed Projects", f"{delayed_projects:,}", "Projects exceeding duration threshold")
    with cols[3]:
        anomaly_count = len(anomalies)
        metric_card("Statistical Anomalies", f"{anomaly_count:,}", "Detected by ML")
    st.markdown("</div>", unsafe_allow_html=True)

    # Explain red flags briefly
    st.markdown("#### Overview")
    st.markdown(
        """
    - **High-Value Projects:** Top 10% by ABC  
    - **Suspicious Cost Savings:** Exceeds selected % threshold  
    - **Delayed Projects:** Duration beyond threshold (days)  
    - **Statistical Anomalies:** Outliers detected by Isolation Forest (multivariate)
    """
    )

    st.divider()

    # Analysis tabs (clean layout)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cost Analysis", "Timeline", "Geographic", "Contractors", "Trends"])

    # ----- Cost Analysis -----
    with tab1:
        card_begin()
        st.markdown("### Cost Analysis")
        st.markdown("Examine distribution of cost savings and relationship between ABC and Contract Cost.")
        c1, c2 = st.columns(2)
        with c1:
            if "Cost_Variance_Pct" in analysis_df.columns and analysis_df["Cost_Variance_Pct"].notna().any():
                fig = px.histogram(
                    analysis_df,
                    x="Cost_Variance_Pct",
                    nbins=50,
                    title="Distribution of Cost Savings (%)",
                    labels={"Cost_Variance_Pct": "Cost Savings (%)"}
                )
                fig.add_vline(x=suspicious_savings, line_dash="dash", line_color="red", annotation_text="Threshold")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cost variance data not available.")
        with c2:
            if "ABC" in analysis_df.columns and "ContractCost" in analysis_df.columns:
                fig2 = px.scatter(
                    analysis_df,
                    x="ABC",
                    y="ContractCost",
                    hover_data=["ProjectID", "Contractor"] if "ProjectID" in analysis_df.columns else None,
                    title="Contract Cost vs Approved Budget",
                    color="Cost_Variance_Pct" if "Cost_Variance_Pct" in analysis_df.columns else None,
                    color_continuous_scale="RdYlGn"
                )
                # identity line
                try:
                    fig2.add_trace(
                        go.Scatter(
                            x=[analysis_df["ABC"].min(), analysis_df["ABC"].max()],
                            y=[analysis_df["ABC"].min(), analysis_df["ABC"].max()],
                            mode="lines",
                            line=dict(dash="dash", color="red"),
                            name="ABC = ContractCost",
                        )
                    )
                except Exception:
                    pass
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ABC vs ContractCost data unavailable.")
        card_end()

    # ----- Timeline -----
    with tab2:
        card_begin()
        st.markdown("### Timeline Analysis")
        if "Duration_Days" in analysis_df.columns and analysis_df["Duration_Days"].notna().any():
            d1, d2 = st.columns(2)
            with d1:
                fig_dur = px.histogram(
                    analysis_df,
                    x="Duration_Days",
                    nbins=30,
                    title="Distribution of Project Duration (Days)",
                    labels={"Duration_Days": "Duration (Days)"}
                )
                fig_dur.add_vline(x=duration_threshold, line_dash="dash", line_color="red", annotation_text="Alert")
                st.plotly_chart(fig_dur, use_container_width=True)
            with d2:
                if "CompletionYear" in analysis_df.columns:
                    duration_by_year = analysis_df.groupby("CompletionYear")["Duration_Days"].mean().reset_index()
                    fig_year = px.bar(duration_by_year, x="CompletionYear", y="Duration_Days", title="Avg Duration by Completion Year")
                    st.plotly_chart(fig_year, use_container_width=True)
                else:
                    st.info("CompletionYear not in dataset.")
        else:
            st.info("Duration data not available.")
        card_end()

    # ----- Geographic -----
    with tab3:
        card_begin()
        st.markdown("### Geographic Distribution")
        if all(col in analysis_df.columns for col in ["Latitude", "Longitude"]):
            map_df = analysis_df.dropna(subset=["Latitude", "Longitude"])
            if len(map_df) > 0:
                hover_cols = ["ProjectDescription", "Contractor", "ContractCost"]
                hover_data = {c: True for c in hover_cols if c in map_df.columns}
                hover_data.update({"Latitude": False, "Longitude": False})
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="ProjectID" if "ProjectID" in map_df.columns else None,
                    hover_data=hover_data,
                    color="ContractCost" if "ContractCost" in map_df.columns else None,
                    size="ContractCost" if "ContractCost" in map_df.columns else None,
                    color_continuous_scale="Viridis",
                    zoom=5,
                    height=550
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No valid coordinates to map.")
        else:
            st.info("Latitude/Longitude not available.")
        card_end()

    # ----- Contractors -----
    with tab4:
        card_begin()
        st.markdown("### Contractor Performance")
        if "Contractor" in analysis_df.columns:
            stats = analysis_df.groupby("Contractor").agg({
                "ProjectID": "count",
                "ContractCost": ["sum", "mean"],
                "Duration_Days": "mean",
                "Cost_Variance": "sum"
            }).round(2)
            stats.columns = ["Projects", "Total Cost", "Avg Cost", "Avg Duration", "Total Savings"]
            stats = stats.sort_values("Total Cost", ascending=False).head(20)
            c1, c2 = st.columns(2)
            with c1:
                fig_p = px.bar(stats.reset_index(), x="Contractor", y="Projects", title="Top Contractors by # Projects")
                fig_p.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_p, use_container_width=True)
            with c2:
                fig_c = px.bar(stats.reset_index(), x="Contractor", y="Total Cost", title="Top Contractors by Total Contract Value", color="Total Cost", color_continuous_scale="Blues")
                fig_c.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_c, use_container_width=True)
            st.dataframe(stats, use_container_width=True)
        else:
            st.info("Contractor data not available.")
        card_end()

    # ----- Trends -----
    with tab5:
        card_begin()
        st.markdown("### Temporal Trends")
        if "InfraYear" in analysis_df.columns:
            yearly = analysis_df.groupby("InfraYear").agg({"ProjectID": "count", "ContractCost": "sum"}).reset_index()
            yearly.columns = ["Year", "Projects", "Total Cost"]
            t1, t2 = st.columns(2)
            with t1:
                fig_projects = px.line(yearly, x="Year", y="Projects", markers=True, title="Number of Projects Over Time")
                st.plotly_chart(fig_projects, use_container_width=True)
            with t2:
                fig_cost = px.line(yearly, x="Year", y="Total Cost", markers=True, title="Total Project Cost Over Time")
                st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("Year data not available.")
        card_end()

    st.divider()

    # Detailed project table + search
    st.markdown("## Detailed Project Data")
    st.markdown("Search by description, contractor, or location. Table updates as you type.")

    search_term = st.text_input("Search projects:", placeholder="Type description, contractor, region, or province...")
    if search_term:
        search_cols = ["ProjectDescription", "Contractor", "Region", "Province", "Municipality"]
        mask = pd.Series(False, index=analysis_df.index)
        for col in search_cols:
            if col in analysis_df.columns:
                mask |= analysis_df[col].astype(str).str.contains(search_term, case=False, na=False)
        display_df = analysis_df[mask].copy()
        st.success(f"Found {len(display_df)} match(es) from {len(analysis_df)} projects")
    else:
        display_df = analysis_df.copy()
        st.info(f"Showing {len(display_df)} projects")

    # Format display columns
    display_cols = ["ProjectID","ProjectDescription","Region","Province","Contractor","ABC","ContractCost","Cost_Variance_Pct","Duration_Days","StartDate","CompletionDateActual"]
    available_display_cols = [c for c in display_cols if c in display_df.columns]
    df_for_display = display_df[available_display_cols].copy()
    if "ABC" in df_for_display.columns:
        df_for_display["ABC"] = df_for_display["ABC"].apply(lambda x: f"₱{x:,.2f}" if pd.notna(x) else "N/A")
    if "ContractCost" in df_for_display.columns:
        df_for_display["ContractCost"] = df_for_display["ContractCost"].apply(lambda x: f"₱{x:,.2f}" if pd.notna(x) else "N/A")
    if "Cost_Variance_Pct" in df_for_display.columns:
        df_for_display["Cost_Variance_Pct"] = df_for_display["Cost_Variance_Pct"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    st.dataframe(df_for_display, use_container_width=True, height=420)

    # Anomalies table (if any)
    if len(anomalies) > 0:
        st.divider()
        st.markdown("## Projects flagged as Statistical Anomalies")
        st.markdown("Anomalies are statistical outliers and need further investigation (not proof of wrongdoing).")
        cols_show = ["ProjectID","ProjectDescription","Contractor","Region","Province","ABC","ContractCost","Cost_Variance_Pct","Duration_Days"]
        cols_show = [c for c in cols_show if c in anomalies.columns]
        anom_display = anomalies[cols_show].copy()
        if "ABC" in anom_display.columns:
            anom_display["ABC"] = anom_display["ABC"].apply(lambda x: f"₱{x:,.2f}" if pd.notna(x) else "N/A")
        if "ContractCost" in anom_display.columns:
            anom_display["ContractCost"] = anom_display["ContractCost"].apply(lambda x: f"₱{x:,.2f}" if pd.notna(x) else "N/A")
        if "Cost_Variance_Pct" in anom_display.columns:
            anom_display["Cost_Variance_Pct"] = anom_display["Cost_Variance_Pct"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        st.dataframe(anom_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Disclaimer:** Anomalies indicate statistical deviation and should be examined with context.")

elif st.session_state.page == "About":
    
    st.title("About This Dashboard")

    st.markdown("""
    ### Purpose  
    This dashboard promotes **transparency and accountability** in public infrastructure spending.
    It uses data-driven analysis to detect patterns, irregularities, and statistical anomalies that may indicate risks and require further investigation.


    ---

    ### Methodology  

    #### **1. Anomaly Detection (Isolation Forest)**
    - Identifies statistical outliers in multidimensional project data  
    - Analyzed features:
        * ABC (Approved Budget for the Contract)  
        * Contract Cost  
        * Project Duration  
        * Cost Variance %  
    - **Contamination Rate:** 10% (assumes 10% of data may be anomalous)

    ---

    #### **2. Red Flag Indicators**
    - **High-Value Projects:** Top 10% by ABC  
    - **Suspicious Cost Savings:** High Cost Variance % (too big or negative)  
    - **Delayed Projects:** Duration exceeds acceptable thresholds

    ---

    #### **3. Statistical Metrics**
    - Cost Variance  
    - Cost Variance %  
    - Duration (Days)  
    - Cost Per Day  

    ---

    ### Disclaimer  
    Detected anomalies represent **statistical deviations**, not proof of wrongdoing.  
    They should be interpreted with **context** and require **manual verification**.
    
    ⚠️ Some provinces may appear under an incorrect region due to inconsistencies in the source dataset.

    ---

    ### Data Source  
    **bettergov.ph**  
    _Last updated: **2024**_
    """)
