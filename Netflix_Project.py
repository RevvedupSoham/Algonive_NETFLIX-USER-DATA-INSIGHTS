import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


# Directory Setup
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


# 1. LOAD DATA
print("\nLoading dataset...")
df = pd.read_csv("netflix_titles.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# 2. DATA CLEANING & PREPROCESSING
print("\nCleaning data...")

df.drop_duplicates(inplace=True)

df['country'] = df['country'].fillna("Unknown")
df['director'] = df['director'].fillna("Not Available")
df['cast'] = df['cast'].fillna("Not Available")
df['rating'] = df['rating'].fillna("Not Rated")
df['description'] = df['description'].fillna("")

df.dropna(subset=['title', 'type'], inplace=True)

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

def clean_duration(x):
    if "min" in str(x):
        return int(x.replace(" min", ""))
    return np.nan

df['duration_mins'] = df['duration'].apply(clean_duration)
df['num_genres'] = df['listed_in'].apply(lambda x: len(str(x).split(',')))
df['content_age'] = 2026 - df['release_year']

print("After Cleaning Shape:", df.shape)


# 3. EXPLORATORY DATA ANALYSIS
print("\nGenerating EDA plots...")

plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df)
plt.title("Movies vs TV Shows")
plt.tight_layout()
plt.savefig("outputs/plots/movies_vs_tv.png")
plt.close()

plt.figure(figsize=(10,6))
df['listed_in'].str.split(', ').explode().value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Genres")
plt.tight_layout()
plt.savefig("outputs/plots/top_genres.png")
plt.close()

plt.figure(figsize=(10,6))
df['year_added'].value_counts().sort_index().plot(kind='line')
plt.title("Content Added Over Years")
plt.tight_layout()
plt.savefig("outputs/plots/content_over_years.png")
plt.close()

plt.figure(figsize=(10,6))
df['country'].value_counts().head(10).plot(kind='bar')
plt.title("Top Content Producing Countries")
plt.tight_layout()
plt.savefig("outputs/plots/top_countries.png")
plt.close()


# FEATURE ENGINEERING
features = df[['duration_mins', 'release_year', 'num_genres', 'content_age']]
features = features.fillna(features.mean())

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# CLUSTERING - KMEANS
print("\nRunning KMeans clustering...")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_features)

sil_score = silhouette_score(scaled_features, df['cluster'])

print("Cluster Distribution:\n", df['cluster'].value_counts())
print("Silhouette Score:", round(sil_score, 3))

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['release_year'],
    y=df['duration_mins'],
    hue=df['cluster'],
    palette='viridis'
)
plt.title("Content Clustering")
plt.xlabel("Release Year")
plt.ylabel("Duration (mins)")
plt.tight_layout()
plt.savefig("outputs/plots/clustering.png")
plt.close()


# 6. TRADITIONAL TF-IDF RECOMMENDER
print("\nBuilding TF-IDF recommender...")

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df[['title', 'type', 'listed_in']].iloc[movie_indices]


# DEEP LEARNING AUTOENCODER
print("\nTraining deep learning autoencoder...")

tfidf_small = tfidf_matrix[:, :2000].toarray()

input_dim = tfidf_small.shape[1]
encoding_dim = 64

input_layer = Input(shape=(input_dim,))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
bottleneck = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(bottleneck)
decoded = Dense(512, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

autoencoder.compile(optimizer=Adam(0.001), loss='mse')

autoencoder.fit(
    tfidf_small, tfidf_small,
    epochs=15,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)


# MODEL SAVING
autoencoder.save("outputs/models/autoencoder_model.keras")
encoder_model.save("outputs/models/encoder_model.keras")
content_embeddings = encoder_model.predict(tfidf_small)
np.save("outputs/models/content_embeddings.npy", content_embeddings)


# 8. DEEP LEARNING RECOMMENDER
deep_sim = cosine_similarity(content_embeddings, content_embeddings)
def deep_recommend(title, n=5):
    idx = indices[title]
    sim_scores = list(enumerate(deep_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df[['title', 'type', 'listed_in']].iloc[movie_indices]


# SAVE CLEAN DATA
df.to_csv("outputs/netflix_cleaned_dataset.csv", index=False)


# FINAL INSIGHTS
print("\nKEY INSIGHTS")
print("------------")
print("Total Titles:", df.shape[0])
print("Movies:", df[df['type']=='Movie'].shape[0])
print("TV Shows:", df[df['type']=='TV Show'].shape[0])
print("Top Genre:", df['listed_in'].str.split(', ').explode().value_counts().idxmax())
print("Top Producing Country:", df['country'].value_counts().idxmax())
print("\nAll plots saved in: outputs/plots/")
print("All models saved in: outputs/models/")
print("Project execution complete.")