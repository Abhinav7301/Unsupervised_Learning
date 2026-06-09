import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

from models.hierarchical import HierarchicalClustering

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌳",
    layout="wide"
)

# ---------------------------------
# CSS
# ---------------------------------

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
color:#10b981;
text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.sub-title{
text-align:center;
font-size:18px;
color:#cbd5e1;
}

[data-testid="metric-container"]{
background:linear-gradient(135deg, #1e293b, #334155);
border-radius:12px;
padding:15px;
border:2px solid #10b981;
box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
}

[data-testid="stSidebar"]{
background:#1e293b;
border-right: 2px solid #10b981;
}

.stTabs [data-baseweb="tab"] {
    font-size:16px;
    font-weight:600;
    color:#cbd5e1;
}

.stDownloadButton button {
    background: linear-gradient(135deg, #10b981, #059669);
    color:white;
    border-radius:8px;
    font-weight:bold;
    border:2px solid #10b981;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# HEADER
# ---------------------------------

st.markdown("""
<div class='main-title'>
🌳 Hierarchical Clustering Dashboard
</div>
<div class='sub-title'>
Agglomerative Clustering on Iris Dataset
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------
# LOAD DATA
# ---------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("⚙ Settings")

clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    10,
    3
)

# ---------------------------------
# MODEL
# ---------------------------------

model = HierarchicalClustering(clusters)

result = model.fit(df)

X = result["X_scaled"]
labels = result["labels"]
score = result["score"]

df["Cluster"] = labels

# ---------------------------------
# KPI
# ---------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Flowers", len(df))
c2.metric("Features", 4)
c3.metric("Clusters", clusters)
c4.metric("Silhouette", round(score,3))

st.divider()

# ---------------------------------
# TABS
# ---------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Dataset",
"Dendrogram",
"Clusters",
"Export"
]
)

# ---------------------------------
# DATASET
# ---------------------------------

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

# ---------------------------------
# DENDROGRAM
# ---------------------------------

with tab2:

    st.subheader("Hierarchical Dendrogram")

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    linked = linkage(
        X,
        method='ward'
    )

    dendrogram(
        linked,
        ax=ax,
        color_threshold=0
    )

    plt.title("Dendrogram", fontsize=14, color="#cbd5e1")
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#cbd5e1")
    ax.spines["bottom"].set_color("#10b981")
    ax.spines["left"].set_color("#10b981")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig)

# ---------------------------------
# CLUSTERS
# ---------------------------------

with tab3:

    fig = px.scatter(
        x=X[:,0],
        y=X[:,1],
        color=labels.astype(str),
        title="Cluster Visualization",
        color_discrete_sequence=[
            "#10b981", "#06b6d4", "#f59e0b", "#ef4444", 
            "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
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

    counts = (
        df["Cluster"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Cluster",
        "Count"
    ]

    pie = px.pie(
        counts,
        names="Cluster",
        values="Count",
        hole=0.5,
        color_discrete_sequence=[
            "#10b981", "#06b6d4", "#f59e0b", "#ef4444", 
            "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
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

# ---------------------------------
# EXPORT
# ---------------------------------

with tab4:

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Download Dataset",
        csv,
        "hierarchical_clusters.csv",
        "text/csv"
    )
