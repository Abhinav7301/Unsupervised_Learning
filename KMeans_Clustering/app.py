import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from models.kmeans import KMeansClustering

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Iris K-Means Dashboard",
    page_icon="🌸",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
}

/* Header */
.main-title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#06b6d4;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.sub-title {
    text-align:center;
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:20px;
    font-weight:bold;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1e293b;
    border-right: 2px solid #06b6d4;
}

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b, #334155);
    border:2px solid #06b6d4;
    padding:15px;
    border-radius:12px;
    box-shadow:0 4px 6px rgba(6, 182, 212, 0.2);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #1e293b;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size:16px;
    font-weight:600;
    color:#cbd5e1;
}

/* Buttons */
.stDownloadButton button {
    background: linear-gradient(135deg, #06b6d4, #0891b2);
    color:white;
    border-radius:8px;
    font-weight:bold;
    border:2px solid #06b6d4;
}

.stButton button {
    background: linear-gradient(135deg, #06b6d4, #0891b2);
    color:white;
    border-radius:8px;
    font-weight:bold;
    border:2px solid #06b6d4;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="main-title">
        🌸 Iris Flower Clustering Dashboard
    </div>
    <div class="sub-title">
        K-Means Clustering using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# LOAD DATA
# ==================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ Dashboard Settings")

clusters = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
    value=3
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Dataset: Iris Dataset
    
    Features:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width
    """
)

# ==================================================
# MODEL
# ==================================================

model = KMeansClustering(clusters)

result = model.fit(df)

X = result["X_scaled"]
labels = result["labels"]
score = result["score"]
centers = result["centers"]

df["Cluster"] = labels

# ==================================================
# KPI SECTION
# ==================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "🌺 Flowers",
        len(df)
    )

with c2:
    st.metric(
        "📊 Features",
        4
    )

with c3:
    st.metric(
        "🎯 Clusters",
        clusters
    )

with c4:
    st.metric(
        "⭐ Silhouette Score",
        round(score, 3)
    )

st.divider()

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 Dataset",
        "📊 Visualization",
        "📈 Analytics",
        "⬇️ Export"
    ]
)

# ==================================================
# DATASET TAB
# ==================================================

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(15),
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ==================================================
# VISUALIZATION TAB
# ==================================================

with tab2:

    st.subheader("Cluster Visualization")

    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=labels.astype(str),
        title="K-Means Clustering Result",
        color_discrete_sequence=[
            "#06b6d4", "#10b981", "#f59e0b", "#ef4444", 
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

    st.subheader("Cluster Centers")

    centers_df = pd.DataFrame(
        centers,
        columns=[
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ]
    )

    st.dataframe(
        centers_df,
        use_container_width=True
    )

# ==================================================
# ANALYTICS TAB
# ==================================================

with tab3:

    col1, col2 = st.columns(2)

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Count"
    ]

    with col1:

        pie = px.pie(
            cluster_counts,
            values="Count",
            names="Cluster",
            hole=0.5,
            title="Cluster Distribution",
            color_discrete_sequence=[
                "#06b6d4", "#10b981", "#f59e0b", "#ef4444", 
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

    with col2:

        bar = px.bar(
            cluster_counts,
            x="Cluster",
            y="Count",
            color="Cluster",
            title="Cluster Sizes",
            color_discrete_sequence=[
                "#06b6d4", "#10b981", "#f59e0b", "#ef4444", 
                "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
                "#6366f1", "#84cc16"
            ]
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

# ==================================================
# EXPORT TAB
# ==================================================

with tab4:

    st.subheader("Download Clustered Dataset")

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="kmeans_clustered_data.csv",
        mime="text/csv"
    )

    st.success(
        "Dataset is ready to download."
    )
