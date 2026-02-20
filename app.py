import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# PAGE CONFIG
st.set_page_config(
    page_title="Algonive | AI OTT Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color:#f5f7fb;
}

.main-title {
    font-size:60px;
    font-weight:800;
    text-align:center;
    color:#111827;
}

.sub-title {
    font-size:22px;
    text-align:center;
    color:#6b7280;
}

.metric-card {
    background:white;
    padding:24px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 10px 30px rgba(0,0,0,0.06);
}

.metric-card h2 {
    font-size:44px;
    color:#dc2626;
}

.loader {
  border:6px solid #e5e7eb;
  border-top:6px solid #dc2626;
  border-radius:50%;
  width:60px;
  height:60px;
  animation:spin 1s linear infinite;
  margin:auto;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)


# INTRO ANIMATION
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    st.markdown("<div class='main-title'>Algonive</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>NETFLIX / OTT USER DATA INSIGHTS</div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='loader'></div>", unsafe_allow_html=True)
    time.sleep(3)
    st.session_state.intro_done = True
    st.rerun()


# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("outputs/netflix_cleaned_dataset.csv")

df = load_data()


# LOAD MODELS
@st.cache_resource
def load_models():
    encoder = load_model("outputs/models/encoder_model.keras")
    embeddings = np.load("outputs/models/content_embeddings.npy")
    return encoder, embeddings

encoder, embeddings = load_models()


# USER PROFILE MEMORY
if "user_likes" not in st.session_state:
    st.session_state.user_likes = []

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Overview", "Visual Analytics", "Semantic Search", "Personalized AI"])


# OVERVIEW
if menu == "Overview":
    c1,c2,c3,c4 = st.columns(4)

    c1.markdown(f"<div class='metric-card'><h2>{len(df)}</h2><p>Total Titles</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h2>{df['type'].value_counts()['Movie']}</h2><p>Movies</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h2>{df['type'].value_counts()['TV Show']}</h2><p>TV Shows</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h2>{df['listed_in'].str.split(', ').explode().value_counts().idxmax()}</h2><p>Top Genre</p></div>", unsafe_allow_html=True)


# VISUAL ANALYTICS
elif menu == "Visual Analytics":

    col1,col2 = st.columns(2)
    fig1 = px.pie(df, names='type', hole=0.45)
    fig2 = px.bar(df['listed_in'].str.split(', ').explode().value_counts().head(10))

    with col1: st.plotly_chart(fig1, use_container_width=True)
    with col2: st.plotly_chart(fig2, use_container_width=True)


# SEMANTIC SEARCH
elif menu == "Semantic Search":

    st.markdown("## Semantic Movie Search")

    query = st.text_input("Describe what you want to watch:",
        placeholder="Example: emotional romantic movie with sad ending...")

    if query:

        # Encode query using trained encoder
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
        tfidf_matrix = tfidf.fit_transform(df['description'].fillna(''))

        query_vec = tfidf.transform([query]).toarray()
        query_embed = encoder.predict(query_vec)

        sim = cosine_similarity(query_embed, embeddings).flatten()
        top_idx = sim.argsort()[-6:-1][::-1]

        st.success("Best matches for your mood...")
        st.dataframe(df.iloc[top_idx][['title','type','listed_in','release_year']], use_container_width=True)


# PERSONALIZED RECOMMENDER
elif menu == "Personalized AI":

    st.markdown("## Personalized AI Recommendation")

    liked = st.multiselect(
        "Select movies / shows you liked:",
        options=sorted(df['title'].unique())
    )

    if st.button("Build My Taste Profile") and liked:

        indices = df[df['title'].isin(liked)].index
        user_vector = embeddings[indices].mean(axis=0).reshape(1,-1)

        sim = cosine_similarity(user_vector, embeddings).flatten()
        top_idx = sim.argsort()[-8:][::-1]

        st.success("AI-curated picks just for you...")

        st.dataframe(df.iloc[top_idx][['title','type','listed_in','release_year']], use_container_width=True)


# FOOTER
st.markdown("---")
st.markdown("<center>Algonive | AI OTT Recommendation Platform</center>", unsafe_allow_html=True)