import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from models.gmm import GMMClustering

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="GMM Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------
# CSS
# ----------------------------------

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
color:#ec4899;
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
border:2px solid #ec4899;
box-shadow:0 4px 6px rgba(236, 72, 153, 0.2);
}

[data-testid="stSidebar"]{
background:#1e293b;
border-right: 2px solid #ec4899;
}

.stTabs [data-baseweb="tab"] {
    font-size:16px;
    font-weight:600;
    color:#cbd5e1;
}

.stDownloadButton button {
    background: linear-gradient(135deg, #ec4899, #db2777);
    color:white;
    border-radius:8px;
    font-weight:bold;
    border:2px solid #ec4899;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown("""
<div class='main-title'>
📈 Gaussian Mixture Model Dashboard
</div>

<div class='sub-title'>
Probabilistic Clustering using Iris Dataset
</div>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------------
# LOAD DATASET
# ----------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("⚙ Settings")

components = st.sidebar.slider(
    "Number of Components",
    2,
    10,
    3
)

# ----------------------------------
# MODEL
# ----------------------------------

model = GMMClustering(components)

result = model.fit(df)

X = result["X_scaled"]
labels = result["labels"]
score = result["score"]
means = result["means"]
probabilities = result["probabilities"]

df["Cluster"] = labels

# ----------------------------------
# KPI SECTION
# ----------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Flowers", len(df))
c2.metric("Features", 4)
c3.metric("Components", components)
c4.metric("Silhouette", round(score,3))

st.divider()

# ----------------------------------
# TABS
# ----------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Dataset",
"Clusters",
"Analytics",
"Export"
]
)

# ----------------------------------
# DATASET
# ----------------------------------

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(15),
        use_container_width=True
    )

    st.subheader("Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ----------------------------------
# CLUSTERS
# ----------------------------------

with tab2:

    fig = px.scatter(
        x=X[:,0],
        y=X[:,1],
        color=labels.astype(str),
        title="GMM Cluster Visualization",
        color_discrete_sequence=[
            "#ec4899", "#06b6d4", "#10b981", "#f59e0b", 
            "#8b5cf6", "#ef4444", "#14b8a6", "#f97316",
            "#6366f1", "#84cc16"
        ]
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

    st.subheader("Gaussian Means")

    means_df = pd.DataFrame(
        means,
        columns=df.columns[:-1]
    )

    st.dataframe(
        means_df,
        use_container_width=True
    )

# ----------------------------------
# ANALYTICS
# ----------------------------------

with tab3:

    cluster_df = (
        df["Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_df.columns = [
        "Cluster",
        "Count"
    ]

    pie = px.pie(
        cluster_df,
        names="Cluster",
        values="Count",
        hole=0.5,
        title="Cluster Distribution",
        color_discrete_sequence=[
            "#ec4899", "#06b6d4", "#10b981", "#f59e0b", 
            "#8b5cf6", "#ef4444", "#14b8a6", "#f97316",
            "#6366f1", "#84cc16"
        ]
    )

    pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        font=dict(color="#cbd5e1")
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    st.subheader(
        "Cluster Membership Probabilities"
    )

    prob_df = pd.DataFrame(
        probabilities[:10]
    )

    st.dataframe(
        prob_df,
        use_container_width=True
    )

# ----------------------------------
# EXPORT
# ----------------------------------

with tab4:

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Download Clustered Dataset",
        csv,
        "gmm_clusters.csv",
        "text/csv"
    )
