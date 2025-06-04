# %%
# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Konfigurasi visualisasi
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# %% [markdown]
# ## Load Datasets

# %%
# Load semua dataset
movies_df = pd.read_csv('datasets/movies.csv')
ratings_df = pd.read_csv('datasets/ratings.csv')
tags_df = pd.read_csv('datasets/tags.csv')
links_df = pd.read_csv('datasets/links.csv')

# %% [markdown]
# ## Data Understanding

# %%
# Tampilkan 5 baris pertama untuk masing-masing
print("Movies:")
print(movies_df.head())
print("\nRatings:")
print(ratings_df.head())
print("\nTags:")
print(tags_df.head())
print("\nLinks:")
print(links_df.head())

# %% [markdown]
# ### Ukuran dan missing value

# %%
# Cek informasi dataset dan missing value
print("Movies Info:")
print(movies_df.info())

print("\nRatings Info:")
print(ratings_df.info())

print("\nTags Info:")
print(tags_df.info())

print("\nLinks Info:")
print(links_df.info())


# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# ### Memeriksa Jumlah Rating

# %%
ratings = pd.read_csv("datasets/ratings.csv")

# Jumlah rating per user
ratings_per_user = ratings.groupby("userId").size()

# Jumlah rating per movie
ratings_per_movie = ratings.groupby("movieId").size()

# Visualisasi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(ratings_per_user, bins=50)
plt.title("Distribusi Jumlah Rating per User")
plt.xlabel("Jumlah Rating")
plt.ylabel("Jumlah User")

plt.subplot(1, 2, 2)
sns.histplot(ratings_per_movie, bins=50)
plt.title("Distribusi Jumlah Rating per Movie")
plt.xlabel("Jumlah Rating")
plt.ylabel("Jumlah Movie")

plt.tight_layout()
plt.show() 
# Tampilkan ringkasan statistik jumlah rating per user dan per movie
print("Statistik Jumlah Rating per User:")
print(ratings_per_user.describe())
print("\nStatistik Jumlah Rating per Movie:")
print(ratings_per_movie.describe())

# %% [markdown]
# ### Distribusi Rating

# %%
# Plot distribusi rating
sns.histplot(ratings_df['rating'], bins=10, kde=True)
plt.title('Distribusi Rating')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.show()
# Tampilkan jumlah masing-masing rating (1,2,3,4,5)
rating_counts = ratings_df['rating'].value_counts().sort_index()
for rating in range(1, 6):
    count = rating_counts.get(rating, 0)
    print(f"Jumlah rating {rating}: {count}")

# %% [markdown]
# ### Film dengan Rating Terbanyak

# %%
# Hitung jumlah rating per film
film_popularitas = ratings_df.groupby('movieId')['rating'].count().sort_values(ascending=False).head(10)

# Gabungkan dengan judul
top_films = pd.merge(film_popularitas.reset_index(), movies_df, on='movieId')
top_films.columns = ['movieId', 'Jumlah Rating', 'title', 'genres']

# Plot
sns.barplot(data=top_films, y='title', x='Jumlah Rating', palette='viridis')
plt.title('10 Film dengan Rating Terbanyak')
plt.xlabel('Jumlah Rating')
plt.ylabel('Judul Film')
plt.show()
# Tampilkan tabel 10 film dengan rating terbanyak
print("10 Film dengan Rating Terbanyak:")
print(top_films[['title', 'Jumlah Rating']].to_string(index=False))

# %% [markdown]
# ## Data Preparation (Content-Based Filtering)

# %%
# Load data film
movies = pd.read_csv('datasets/movies.csv')

# Lihat beberapa data awal
print(movies.head())


# %% [markdown]
# ### Bersihkan kolom genres

# %%
# Ganti '|' menjadi spasi agar bisa diproses oleh TF-IDF
movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)


# %% [markdown]
# ### TF-IDF Vectorization

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi dan transformasi TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])

# Simpan feature names
tfidf_feature_names = tfidf.get_feature_names_out()

# Lihat dimensi matrix
print("TF-IDF Matrix shape:", tfidf_matrix.shape)


# %% [markdown]
# ### Menyimpan Data yang Dibutuhkan

# %%
# Simpan dataframe untuk kebutuhan model nanti
movie_ids = movies['movieId']
movie_titles = movies['title']


# %% [markdown]
# ## Data Preparation (Collaborative Filtering)

# %%
ratings = pd.read_csv('datasets/ratings.csv')

# Tampilkan 5 baris pertama
print(ratings.head())


# %% [markdown]
# ### Buat user-item matrix

# %%
# Buat matriks pengguna x film (nilai = rating)
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Tampilkan ukuran dan contoh matriks
print("User-Item Matrix shape:", user_item_matrix.shape)
user_item_matrix.head()


# %%
# Ganti NaN dengan 0
user_item_matrix_filled = user_item_matrix.fillna(0)


# %% [markdown]
# ## Modeling - Content-Based Filtering

# %% [markdown]
# ### Hitung Cosine Similarity antar film

# %%
from sklearn.metrics.pairwise import linear_kernel

# Hitung cosine similarity dari TF-IDF matrix genre
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Simpan index berdasarkan judul
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


# %% [markdown]
# ### Fungsi untuk mendapatkan rekomendasi

# %%
def get_recommendations(title, cosine_sim=cosine_sim, num_recommendations=10):
    # Dapatkan index dari judul film
    idx = indices[title]

    # Dapatkan pasangan (index, similarity) dan urutkan berdasarkan similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ambil top N film (tidak termasuk dirinya sendiri)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Ambil index film
    movie_indices = [i[0] for i in sim_scores]

    # Kembalikan daftar judul
    return movies['title'].iloc[movie_indices]


# %% [markdown]
# ### Hasil penggunaan — Rekomendasi untuk satu film

# %%
# Contoh: Rekomendasikan film mirip dengan "Toy Story (1995)"
recommended_movies = get_recommendations("Toy Story (1995)")
print("Rekomendasi film mirip dengan 'Toy Story (1995)':")
print(recommended_movies)
# Visualisasi tabel rekomendasi
import matplotlib.pyplot as plt

# Buat DataFrame dari hasil rekomendasi
recommended_df = pd.DataFrame({'Rekomendasi': recommended_movies.values})

# Tampilkan tabel
print("\nTabel Rekomendasi Film:")
display(recommended_df)

# %% [markdown]
# ## Modeling - Collaborative Filtering (Item-based)

# %% [markdown]
# ###  Membuat matrix user-item rating

# %%
# Membuat pivot table userId sebagai baris, movieId sebagai kolom, dan rating sebagai nilai
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)


# %% [markdown]
# ### Hitung kemiripan antar item (movie)

# %%
from sklearn.metrics.pairwise import cosine_similarity

# Hitung cosine similarity antar item (film)
item_similarity = cosine_similarity(user_movie_ratings.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)


# %% [markdown]
# ### Fungsi rekomendasi item-based CF

# %%
def get_item_based_recommendations(movie_id, user_rating=5, top_n=10):
    # Ambil skor kemiripan film lain dengan film yang diberikan
    sim_scores = item_similarity_df[movie_id]

    # Kalikan dengan rating user pada film tersebut
    sim_scores = sim_scores * user_rating

    # Urutkan film berdasarkan skor tertinggi, kecuali film itu sendiri
    sim_scores = sim_scores.sort_values(ascending=False)

    # Keluarkan film yang sama (movie_id)
    sim_scores = sim_scores.drop(movie_id, errors='ignore')

    # Ambil top_n film
    top_movies = sim_scores.head(top_n).index

    # Kembalikan judul film sesuai movieId
    return movies[movies['movieId'].isin(top_movies)]['title']


# %% [markdown]
# ### Hasil penggunaan

# %%
# Misal user memberi rating 5 untuk film dengan movieId 1 (Toy Story)
recommended_items = get_item_based_recommendations(1, user_rating=5, top_n=10)
print("Rekomendasi film berdasarkan rating item Toy Story (movieId=1):")
print(recommended_items)
# Tampilkan urutan rekomendasi film dengan nomor saja (tanpa visual/bar/warna)
print("\nUrutan Rekomendasi Film (Item-based CF) untuk Toy Story (1995):")
for i, title in enumerate(recommended_items.values, 1):
    print(f"{i}. {title}")


# %% [markdown]
# ## Evaluation Content-Based Filtering (Precision)

# %%
# Contoh evaluasi precision untuk satu user
# Misal, user_id = 111, dan kita tahu film apa saja yang pernah diberi rating tinggi oleh user ini
user_id = 111
user_rated_movies = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4)]['movieId']
user_rated_titles = movies_df[movies_df['movieId'].isin(user_rated_movies)]['title']

# Pilih salah satu film yang pernah disukai user
if not user_rated_titles.empty:
    test_title = user_rated_titles.iloc[0]
    recommended_titles = get_recommendations(test_title, num_recommendations=10)
    # Hitung precision: berapa dari rekomendasi yang juga pernah diberi rating tinggi oleh user
    relevant = recommended_titles.isin(user_rated_titles).sum()
    precision = relevant / len(recommended_titles)
    print(f"Precision untuk user {user_id} dengan query '{test_title}': {precision:.2f}")
else:
    print("User tidak memiliki film dengan rating tinggi untuk evaluasi.")

# %%
# Daftar userId yang ingin dievaluasi
user_ids = [4, 100, 20, 30, 23, 50, 60, 70, 80, 90, 111]

precision_results = []

for user_id in user_ids:
    user_rated_movies = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4)]['movieId']
    user_rated_titles = movies_df[movies_df['movieId'].isin(user_rated_movies)]['title']
    if not user_rated_titles.empty:
        test_title = user_rated_titles.iloc[0]
        recommended_titles = get_recommendations(test_title, num_recommendations=10)
        relevant = recommended_titles.isin(user_rated_titles).sum()
        precision = relevant / len(recommended_titles)
        precision_results.append({
            'userId': user_id,
            'query_film': test_title,
            'precision': precision
        })
    else:
        precision_results.append({
            'userId': user_id,
            'query_film': '-',
            'precision': None
        })

precision_df = pd.DataFrame(precision_results)
print(precision_df)

# %% [markdown]
# ### Visualisasi Precision Beberapa User

# %%
# Filter hanya user yang memiliki nilai precision (bukan None)
plot_df = precision_df.dropna(subset=['precision'])

plt.figure(figsize=(8, 5))
plt.bar(plot_df['userId'].astype(str), plot_df['precision'], color='skyblue')
plt.xlabel('User ID')
plt.ylabel('Precision')
plt.title('Precision Content-Based Filtering untuk Beberapa User')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# ## Evaluasi Collaborative Filtering (RMSE)

# %% [markdown]
# ### Split Data

# %%
# Split data menjadi train dan test (misal, 80% train, 20% test)
train, test = train_test_split(ratings_df, test_size=0.2, random_state=42)

# %% [markdown]
# ### Train

# %%
# Buat user-item matrix dari data train
train_matrix = train.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Hitung similarity matrix dari data train
item_similarity_train = cosine_similarity(train_matrix.T)
item_similarity_train_df = pd.DataFrame(item_similarity_train, index=train_matrix.columns, columns=train_matrix.columns)

def predict_rating(user_id, movie_id, user_movie_matrix, similarity_matrix):
    if movie_id not in similarity_matrix.columns or user_id not in user_movie_matrix.index:
        return np.nan
    # Ambil rating user untuk semua film
    user_ratings = user_movie_matrix.loc[user_id]
    # Ambil similarity film ini dengan semua film lain
    sim_scores = similarity_matrix[movie_id]
    # Hanya gunakan film yang pernah dirating user
    mask = user_ratings > 0
    if mask.sum() == 0:
        return np.nan
    sim_scores = sim_scores[mask]
    user_ratings = user_ratings[mask]
    if sim_scores.sum() == 0:
        return np.nan
    # Prediksi rating sebagai weighted average
    return np.dot(sim_scores, user_ratings) / sim_scores.sum()

# %% [markdown]
# ### Test

# %%
# Prediksi rating untuk data test
y_true = []
y_pred = []
for row in test.itertuples():
    pred = predict_rating(row.userId, row.movieId, train_matrix, item_similarity_train_df)
    if not np.isnan(pred):
        y_true.append(row.rating)
        y_pred.append(pred)

if y_true:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE Collaborative Filtering (item-based): {rmse:.2f}")
else:
    print("Tidak ada prediksi yang dapat dihitung untuk RMSE.")

# %%
# Ambil userId unik dari data test
user_ids_test = test['userId'].unique()

user_rmse_results = []

for user_id in user_ids_test[:10]:  # ambil 10 user pertama (atau ganti sesuai kebutuhan)
    user_test = test[test['userId'] == user_id]
    y_true = []
    y_pred = []
    for row in user_test.itertuples():
        pred = predict_rating(row.userId, row.movieId, train_matrix, item_similarity_train_df)
        if not np.isnan(pred):
            y_true.append(row.rating)
            y_pred.append(pred)
    if y_true:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        user_rmse_results.append({'userId': user_id, 'rmse': rmse})
    else:
        user_rmse_results.append({'userId': user_id, 'rmse': None})

user_rmse_df = pd.DataFrame(user_rmse_results)
print(user_rmse_df)

# %%
# Visualisasi RMSE per user
plot_rmse_df = user_rmse_df.dropna(subset=['rmse'])

plt.figure(figsize=(8, 5))
plt.bar(plot_rmse_df['userId'].astype(str), plot_rmse_df['rmse'], color='salmon')
plt.xlabel('User ID')
plt.ylabel('RMSE')
plt.title('RMSE Collaborative Filtering (item-based) per User')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


