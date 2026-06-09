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
        #FF1493 0%,
        #FFD700 25%,
        #00CED1 50%,
        #FF69B4 75%,
        #32CD32 100%
    );
}

/* Header */
.main-title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#FF1493;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.sub-title {
    text-align:center;
    font-size:18px;
    color:#FFFFFF;
    margin-bottom:20px;
    font-weight:bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #00CED1, #00FFFF);
}

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #FFD700, #FFA500);
    border:3px solid #FF1493;
    padding:15px;
    border-radius:15px;
    box-shadow:0 4px 12px rgba(255,20,147,0.4);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: linear-gradient(135deg, #F0F8FF, #E0FFFF);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size:16px;
    font-weight:600;
    color:#FF1493;
}

/* Buttons */
.stDownloadButton button {
    background: linear-gradient(135deg, #FF1493, #FF69B4);
    color:white;
    border-radius:10px;
    font-weight:bold;
    border:2px solid #FFD700;
}

.stButton button {
    background: linear-gradient(135deg, #00CED1, #00FFFF);
    color:#000;
    border-radius:10px;
    font-weight:bold;
    border:2px solid #FF1493;
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
        color_discrete_sequence=["#FF1493", "#00CED1", "#FFD700", "#FF69B4", "#32CD32", "#FFA500", "#00FF7F", "#FF6347", "#9370DB", "#20B2AA"]
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor="rgba(255,20,147,0.1)",
        plot_bgcolor="rgba(0,206,209,0.1)"
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
            color_discrete_sequence=["#FF1493", "#00CED1", "#FFD700", "#FF69B4", "#32CD32", "#FFA500", "#00FF7F", "#FF6347", "#9370DB", "#20B2AA"]
        )

        pie.update_layout(
            template="plotly_dark"
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
            color_discrete_sequence=["#FF1493", "#00CED1", "#FFD700", "#FF69B4", "#32CD32", "#FFA500", "#00FF7F", "#FF6347", "#9370DB", "#20B2AA"]
        )

        bar.update_layout(
            template="plotly_dark"
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
