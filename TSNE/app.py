import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris

from models.tsne_model import TSNEModel

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="t-SNE Dashboard",
    page_icon="🧠",
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
color:#6366f1;
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
border:2px solid #6366f1;
box-shadow:0 4px 6px rgba(99, 102, 241, 0.2);
}

[data-testid="stSidebar"]{
background:#1e293b;
border-right: 2px solid #6366f1;
}

.stTabs [data-baseweb="tab"] {
    font-size:16px;
    font-weight:600;
    color:#cbd5e1;
}

.stDownloadButton button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color:white;
    border-radius:8px;
    font-weight:bold;
    border:2px solid #6366f1;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown("""
<div class='main-title'>
🧠 t-SNE Dashboard
</div>

<div class='sub-title'>
t-Distributed Stochastic Neighbor Embedding
</div>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------------
# DATASET
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

perplexity = st.sidebar.slider(
    "Perplexity",
    min_value=5,
    max_value=50,
    value=30
)

# ----------------------------------
# MODEL
# ----------------------------------

model = TSNEModel(
    perplexity=perplexity
)

result = model.fit(df)

embedding = result["embedding"]

# ----------------------------------
# KPI
# ----------------------------------

c1, c2, c3 = st.columns(3)

c1.metric("Samples", len(df))
c2.metric("Features", 4)
c3.metric("Perplexity", perplexity)

st.divider()

# ----------------------------------
# TABS
# ----------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Dataset",
"t-SNE Visualization",
"Analytics",
"Export"
]
)

# ----------------------------------
# DATASET
# ----------------------------------

with tab1:

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ----------------------------------
# VISUALIZATION
# ----------------------------------

with tab2:

    tsne_df = pd.DataFrame(
        embedding,
        columns=["Dim1","Dim2"]
    )

    fig = px.scatter(
        tsne_df,
        x="Dim1",
        y="Dim2",
        color="Dim1",
        title="t-SNE Projection",
        color_continuous_scale="Viridis"
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

# ----------------------------------
# ANALYTICS
# ----------------------------------

with tab3:

    st.subheader(
        "Embedded Dataset"
    )

    st.dataframe(
        tsne_df.head(20),
        use_container_width=True
    )

# ----------------------------------
# EXPORT
# ----------------------------------

with tab4:

    csv = tsne_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download t-SNE Dataset",
        csv,
        "tsne_output.csv",
        "text/csv"
    )
