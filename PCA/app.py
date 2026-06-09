import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris

from models.pca_model import PCAModel

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="PCA Dashboard",
    page_icon="📉",
    layout="wide"
)

# ------------------------------------
# CSS
# ------------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#0f172a,
#1e293b
);
}

.main-title{
text-align:center;
font-size:48px;
font-weight:bold;
color:#14b8a6;
text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.sub-title{
text-align:center;
font-size:18px;
color:#cbd5e1;
font-weight:bold;
}

[data-testid="metric-container"]{
background:linear-gradient(135deg, #1e293b, #334155);
border-radius:12px;
padding:15px;
border:2px solid #14b8a6;
box-shadow:0 4px 6px rgba(20, 184, 166, 0.2);
}

[data-testid="stSidebar"]{
background:#1e293b;
border-right: 2px solid #14b8a6;
}

.stTabs [data-baseweb="tab"] {
    font-size:16px;
    font-weight:600;
    color:#cbd5e1;
}

.stDownloadButton button {
    background: linear-gradient(135deg, #14b8a6, #0d9488);
    color:white;
    border-radius:8px;
    font-weight:bold;
    border:2px solid #14b8a6;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------
# HEADER
# ------------------------------------

st.markdown("""
<div class='main-title'>
📉 PCA Dashboard
</div>

<div class='sub-title'>
Principal Component Analysis using Iris Dataset
</div>
""", unsafe_allow_html=True)

st.divider()

# ------------------------------------
# LOAD DATASET
# ------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ------------------------------------
# SIDEBAR
# ------------------------------------

st.sidebar.title("⚙ Settings")

components = st.sidebar.slider(
    "Principal Components",
    2,
    4,
    2
)

# ------------------------------------
# MODEL
# ------------------------------------

model = PCAModel(
    n_components=components
)

result = model.fit(df)

X_pca = result["components"]
variance = result["variance"]

# ------------------------------------
# KPI SECTION
# ------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Samples", len(df))
c2.metric("Original Features", 4)
c3.metric("PCA Components", components)
c4.metric(
    "Variance Explained",
    f"{round(sum(variance)*100,2)}%"
)

st.divider()

# ------------------------------------
# TABS
# ------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Dataset",
"PCA Projection",
"Analytics",
"Export"
]
)

# ------------------------------------
# DATASET
# ------------------------------------

with tab1:

    st.subheader("Dataset")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.subheader("Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ------------------------------------
# PCA VISUALIZATION
# ------------------------------------

with tab2:

    pca_df = pd.DataFrame(
        X_pca,
        columns=[
            f"PC{i+1}"
            for i in range(components)
        ]
    )

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA Projection",
        color="PC1",
        color_continuous_scale="Turbo"
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#cbd5e1")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ------------------------------------
# ANALYTICS
# ------------------------------------

with tab3:

    variance_df = pd.DataFrame({
        "Component":
        [f"PC{i+1}" for i in range(len(variance))],
        "Variance":
        variance
    })

    st.subheader(
        "Explained Variance Ratio"
    )

    st.dataframe(
        variance_df,
        use_container_width=True
    )

    bar = px.bar(
        variance_df,
        x="Component",
        y="Variance",
        color="Variance",
        title="Variance Explained",
        color_continuous_scale="Teal"
    )

    bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#cbd5e1")
    )

    st.plotly_chart(
        bar,
        use_container_width=True
    )

# ------------------------------------
# EXPORT
# ------------------------------------

with tab4:

    export_df = pd.DataFrame(
        X_pca,
        columns=[
            f"PC{i+1}"
            for i in range(components)
        ]
    )

    csv = export_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download PCA Dataset",
        csv,
        "pca_output.csv",
        "text/csv"
    )
