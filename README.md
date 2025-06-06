# MovieLens Recommendation System

###### Disusun oleh : Yoga Samudra

Ini adalah proyek sistem rekomendasi film berbasis MovieLens untuk memenuhi submission kelas *machine learning* terapan. Proyek ini membangun model *machine learning* yang dapat memberikan rekomendasi film kepada pengguna berdasarkan data MovieLens.

## Project Overview

### Latar belakang

Domain proyek ini adalah industri hiburan digital, khususnya layanan streaming film dan sistem rekomendasi film. Di era digital saat ini, pengguna dihadapkan pada ribuan pilihan film, sehingga menemukan film yang sesuai dengan preferensi menjadi tantangan tersendiri.

MovieLens adalah salah satu dataset film paling populer yang digunakan untuk penelitian dan pengembangan sistem rekomendasi. Dataset ini berisi data rating film dari ribuan pengguna terhadap ribuan judul film, lengkap dengan informasi genre, tag, dan metadata lainnya.

Tantangan utama dalam domain ini adalah bagaimana menyediakan rekomendasi film yang relevan dan personal bagi setiap pengguna, sehingga pengalaman menonton menjadi lebih menyenangkan dan efisien.
Model machine learning dapat menganalisis pola rating, preferensi genre, dan perilaku pengguna untuk memberikan rekomendasi film yang sesuai.

Dengan membangun sistem rekomendasi berbasis MovieLens, diharapkan pengguna dapat menemukan film-film yang sesuai dengan minat dan kebutuhannya, serta membantu platform streaming dalam meningkatkan kepuasan dan retensi pengguna.

## Business Understanding

Dalam konteks proyek ini, pemahaman bisnis adalah memahami tujuan dan manfaat yang ingin dicapai oleh pengembangan sistem rekomendasi film berbasis MovieLens.

Beberapa poin penting dalam pemahaman bisnis proyek ini:

1. Meningkatkan pengalaman pengguna: Sistem rekomendasi bertujuan untuk meningkatkan pengalaman pengguna dengan menyediakan rekomendasi film yang personal dan relevan, sehingga pengguna dapat menemukan film yang sesuai dengan minat dan preferensi mereka.

2. Meningkatkan retensi pengguna: Dengan rekomendasi yang akurat dan menarik, pengguna cenderung lebih sering menggunakan platform dan tetap aktif sebagai pelanggan.

3. Meningkatkan engagement dan waktu tonton: Rekomendasi yang tepat dapat meningkatkan jumlah film yang ditonton per pengguna, sehingga meningkatkan engagement dan waktu tonton di platform.

4. Mendukung strategi pemasaran dan promosi: Sistem rekomendasi dapat digunakan untuk mempromosikan film-film baru atau kurang populer kepada pengguna yang berpotensi tertarik.

5. Analisis data pengguna: Dengan menganalisis data rating dan perilaku pengguna, platform dapat memahami tren, preferensi, dan kebutuhan pengguna untuk pengembangan layanan lebih lanjut.

### Problem Statements

Berdasarkan pemahaman bisnis di atas, proyek ini berusaha menjawab beberapa pertanyaan utama berikut dalam konteks sistem rekomendasi film berbasis MovieLens:

1. Bagaimana membangun sistem rekomendasi yang dapat memberikan pengalaman pengguna yang lebih baik dalam mencari dan memilih film sesuai preferensi mereka?

2. Bagaimana meningkatkan retensi pengguna dengan menyediakan rekomendasi film yang relevan dan menarik agar pengguna tetap aktif menggunakan platform?

3. Bagaimana mengidentifikasi preferensi genre, pola rating, dan perilaku pengguna untuk mengoptimalkan penyajian film yang relevan?

4. Bagaimana mengukur dan meningkatkan akurasi sistem rekomendasi, baik dari sisi relevansi rekomendasi maupun prediksi rating film?

5. Bagaimana menganalisis data rating dan interaksi pengguna untuk memahami kebutuhan dan tren dalam konsumsi film?

Dengan fokus pada pertanyaan-pertanyaan di atas, tujuan proyek menjadi lebih terarah dan solusi yang dikembangkan dapat memberikan dampak nyata dalam meningkatkan pengalaman pengguna dan performa platform.

### Goals

Tujuan utama dari proyek MovieLens Recommendation System ini adalah:

1. Meningkatkan pengalaman pengguna dengan menyediakan rekomendasi film yang personal dan relevan, sehingga pengguna dapat lebih mudah menemukan film sesuai minat dan preferensi mereka.
- Keberhasilan tujuan ini diukur melalui nilai precision dari hasil rekomendasi (khususnya pada pendekatan content-based), serta melalui tingkat kepuasan dan keterlibatan pengguna terhadap film yang direkomendasikan.

2. Meningkatkan retensi dan engagement pengguna dengan memberikan rekomendasi yang akurat dan menarik, sehingga pengguna lebih sering kembali dan aktif di platform.
- Pencapaian tujuan ini dapat dilihat dari jumlah interaksi atau film yang ditonton berdasarkan rekomendasi, serta frekuensi kunjungan ulang pengguna ke platform.

3. Mengoptimalkan penyajian film yang relevan dengan menganalisis pola rating, genre favorit, dan perilaku pengguna.
- Efektivitas upaya ini tercermin dari seberapa baik sistem mengenali preferensi genre dan pola rating pengguna, sehingga rekomendasi yang diberikan semakin sesuai dengan kebutuhan masing-masing individu.

4. Mengukur dan meningkatkan akurasi sistem rekomendasi menggunakan pendekatan evaluasi yang sesuai, seperti precision untuk content-based filtering dan RMSE untuk collaborative filtering.
- Tingkat keberhasilan model dalam memprediksi dan merekomendasikan film dievaluasi melalui perhitungan nilai precision dan RMSE pada data uji.

5. Memberikan insight bagi pengembangan platform melalui analisis data pengguna, tren genre, dan pola konsumsi film.
- Dampak dari tujuan ini dapat dilihat dari kemampuan sistem dalam menghasilkan analisis dan visualisasi yang membantu pengelola platform memahami kebutuhan serta tren perilaku pengguna.

Dengan tujuan-tujuan ini, sistem rekomendasi yang dibangun diharapkan dapat memberikan manfaat nyata baik bagi pengguna maupun pengelola platform film.

## Data Understanding

Pada tahap ini dilakukan pemahaman awal terhadap struktur dan isi data MovieLens yang digunakan dalam proyek sistem rekomendasi film. Dataset MovieLens dapat diunduh di: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

Dataset terdiri dari empat file utama:

- **movies.csv**: berisi informasi film
- **ratings.csv**: berisi data rating yang diberikan pengguna ke film
- **tags.csv**: berisi tag/kata kunci yang diberikan pengguna ke film
- **links.csv**: berisi relasi movieId ke id eksternal (imdbId, tmdbId)

#### Penjelasan Variabel pada Setiap Dataset

**movies.csv**
- `movieId`: ID unik untuk setiap film
- `title`: Judul film beserta tahun rilis
- `genres`: Daftar genre yang dimiliki film, dipisahkan dengan tanda '|'

**ratings.csv**
- `userId`: ID unik untuk setiap pengguna
- `movieId`: ID film yang dirating
- `rating`: Nilai rating yang diberikan pengguna (skala 0.5 - 5.0)
- `timestamp`: Waktu rating diberikan (format UNIX timestamp)

**tags.csv**
- `userId`: ID pengguna yang memberikan tag
- `movieId`: ID film yang diberi tag
- `tag`: Kata kunci/tag yang diberikan pengguna
- `timestamp`: Waktu tag diberikan (format UNIX timestamp)

**links.csv**
- `movieId`: ID film pada MovieLens
- `imdbId`: ID film pada database IMDb
- `tmdbId`: ID film pada database TMDb

#### Contoh Data

**movies.csv**
| movieId | title                        | genres                                 |
|---------|------------------------------|----------------------------------------|
| 1       | Toy Story (1995)             | Adventure|Animation|Children|Comedy|Fantasy |
| 2       | Jumanji (1995)               | Adventure|Children|Fantasy           |
| 3       | Grumpier Old Men (1995)      | Comedy|Romance                        |

**ratings.csv**
| userId | movieId | rating | timestamp  |
|--------|---------|--------|------------|
| 1      | 1       | 4.0    | 964982703  |
| 1      | 3       | 4.0    | 964981247  |
| 1      | 6       | 4.0    | 964982224  |

**tags.csv**
| userId | movieId | tag             | timestamp   |
|--------|---------|-----------------|-------------|
| 2      | 60756   | funny           | 1445714994  |
| 2      | 60756   | Highly quotable | 1445714996  |

**links.csv**
| movieId | imdbId  | tmdbId  |
|---------|---------|---------|
| 1       | 114709  | 862.0   |
| 2       | 113497  | 8844.0  |

#### Informasi Struktur dan Tipe Data

- **movies.csv**: 9742 baris, 3 kolom (movieId: int64, title: object, genres: object), tanpa missing value.
- **ratings.csv**: 100836 baris, 4 kolom (userId: int64, movieId: int64, rating: float64, timestamp: int64), tanpa missing value.
- **tags.csv**: 3683 baris, 4 kolom (userId: int64, movieId: int64, tag: object, timestamp: int64), tanpa missing value.
- **links.csv**: 9742 baris, 3 kolom (movieId: int64, imdbId: int64, tmdbId: float64), 8 baris pada kolom tmdbId memiliki missing value.

Secara umum, data MovieLens sudah cukup bersih dan siap untuk dilakukan analisis lebih lanjut. Proses EDA akan dilakukan pada tahap berikutnya untuk menggali pola, distribusi, dan insight dari data.

### Exploratory Data Analysis (EDA)

Pada tahap EDA, dilakukan analisis awal untuk memahami pola dan distribusi data rating pada dataset MovieLens. Berikut adalah beberapa insight utama dari hasil eksplorasi:

#### 1. Jumlah Rating per User
- Rata-rata setiap user memberikan 165 rating, dengan minimum 20 dan maksimum 2698 rating.
- Sebaran jumlah rating per user cukup lebar (std: 269), menunjukkan ada user yang sangat aktif dan ada yang hanya memberi sedikit rating.

| Statistik         | Nilai     |
|------------------|-----------|
| Jumlah User      | 610       |
| Rata-rata Rating | 165.3     |
| Median           | 70.5      |
| Maksimum         | 2698      |
| Minimum          | 20        |

#### 2. Jumlah Rating per Movie
- Rata-rata setiap film mendapat 10 rating, namun sebagian besar film hanya mendapat sedikit rating (median: 3, 75% film mendapat kurang dari 9 rating).
- Ada film yang sangat populer dengan ratusan rating.

| Statistik         | Nilai     |
|------------------|-----------|
| Jumlah Film      | 9724      |
| Rata-rata Rating | 10.4      |
| Median           | 3         |
| Maksimum         | 329       |
| Minimum          | 1         |

#### 3. Distribusi Rating
- Rating paling banyak diberikan pada nilai 4 (26.818), diikuti rating 3 (20.047) dan rating 5 (13.211).
- Rating 1 dan 2 relatif lebih sedikit.

| Rating | Jumlah |
|--------|--------|
| 1      | 2.811  |
| 2      | 7.551  |
| 3      | 20.047 |
| 4      | 26.818 |
| 5      | 13.211 |

#### 4. Film dengan Rating Terbanyak
Berikut adalah 10 film dengan jumlah rating terbanyak:

| Judul Film                                 | Jumlah Rating |
|--------------------------------------------|---------------|
| Forrest Gump (1994)                        | 329           |
| Shawshank Redemption, The (1994)           | 317           |
| Pulp Fiction (1994)                        | 307           |
| Silence of the Lambs, The (1991)           | 279           |
| Matrix, The (1999)                         | 278           |
| Star Wars: Episode IV - A New Hope (1977)  | 251           |
| Jurassic Park (1993)                       | 238           |
| Braveheart (1995)                          | 237           |
| Terminator 2: Judgment Day (1991)          | 224           |
| Schindler's List (1993)                    | 220           |

Insight dari EDA ini menunjukkan adanya ketimpangan distribusi rating baik pada user maupun film, serta kecenderungan user memberikan rating tinggi (4 dan 5). Film-film populer cenderung mendapat rating jauh lebih banyak dibanding film lain.

## Data Preparation

Pada bagian ini berisi tahapan persiapan data sebelum pemodelan. Data diproses agar sesuai dengan kebutuhan masing-masing pendekatan (content-based dan collaborative filtering).

### Untuk Content-Based Filtering

Pada tahap ini, data film dipersiapkan untuk membangun model content-based filtering. Langkah-langkah utama yang dilakukan:

1. **Pembersihan Kolom Genre**
   - Kolom `genres` pada dataset film berisi daftar genre yang dipisahkan dengan tanda '|'. Untuk keperluan ekstraksi fitur, tanda pemisah ini diubah menjadi spasi agar dapat diproses oleh TF-IDF Vectorizer.

2. **Ekstraksi Fitur dengan TF-IDF**
   - Menggunakan TF-IDF Vectorizer untuk mengubah teks genre menjadi representasi numerik (vektor fitur). Setiap film direpresentasikan sebagai vektor berdasarkan genre-nya.
   - Hasil transformasi berupa matriks TF-IDF yang siap digunakan untuk menghitung kemiripan antar film.

3. **Penyimpanan Data yang Dibutuhkan**
   - Menyimpan daftar movieId dan title untuk keperluan rekomendasi dan interpretasi hasil model.

## Untuk Collaborative Filtering

Untuk collaborative filtering, data rating pengguna terhadap film diproses menjadi bentuk matriks user-item. Tahapan yang dilakukan:

1. **Membaca Data Rating**
   - Menggunakan dataset ratings.csv yang berisi userId, movieId, rating, dan timestamp.

2. **Membuat User-Item Matrix**
   - Membentuk matriks dengan baris sebagai userId dan kolom sebagai movieId, serta nilai matriks adalah rating yang diberikan user ke film tersebut.
   - Nilai NaN (film yang belum dirating user) diisi dengan 0 agar matriks dapat digunakan untuk perhitungan kemiripan dan prediksi rating.

Data yang sudah dipersiapkan ini akan digunakan pada tahap modeling baik untuk pendekatan content-based maupun collaborative filtering.

## 8. Modeling

Pada tahap modeling, dua pendekatan utama digunakan untuk membangun sistem rekomendasi film berbasis MovieLens, yaitu Content-Based Filtering dan Collaborative Filtering. Berikut penjelasan proses modeling untuk masing-masing pendekatan:

#### 1. Content-Based Filtering
Pendekatan ini merekomendasikan film kepada pengguna berdasarkan kemiripan konten film, khususnya genre. Proses modeling yang dilakukan:

- **Perhitungan Similarity**: Menggunakan cosine similarity untuk mengukur tingkat kemiripan antar film berdasarkan vektor fitur genre yang telah diproses pada tahap data preparation. Semakin tinggi nilai similarity, semakin mirip kedua film tersebut.
- **Rekomendasi Top-N**: Untuk setiap pengguna, sistem mencari film yang paling mirip dengan film-film yang pernah diberi rating tinggi oleh pengguna tersebut, lalu merekomendasikan Top-N film yang belum pernah ditonton.

Output dari model ini berupa daftar rekomendasi film untuk setiap user, beserta nilai similarity-nya.

| Rekomendasi         | Urutan     |
|------------------|-----------|
| Antz (1998)      | 1       |
| Toy Story 2 (1999) | 2     |
| Adventures of Rocky and Bullwinkle, The (2000)           | 3      |
| Emperor's New Groove, The (2000)         | 4      |
| Monsters, Inc. (2001)          | 5        |
| Wild, The (2006)          | 6        |
| Shrek the Third (2007)          | 7        |
| Tale of Despereaux, The (2008)          | 8        |
| Asterix and the Vikings (Astérix et les Viking) (2006)          | 9        |
| Turbo (2013)          | 10        |


#### 2. Collaborative Filtering
Pada pendekatan Collaborative Filtering (item-based), sistem rekomendasi memanfaatkan pola rating pengguna terhadap film untuk mencari kemiripan antar item (film). Berikut tahapan modeling yang dilakukan:

1. **Membuat User-Item Matrix**
   - Data rating diubah menjadi matriks dengan baris sebagai userId, kolom sebagai movieId, dan nilai berupa rating yang diberikan user ke film. Nilai kosong (film yang belum dirating user) diisi dengan 0.

2. **Menghitung Kemiripan Antar Item**
   - Menggunakan cosine similarity untuk menghitung tingkat kemiripan antar film berdasarkan pola rating dari seluruh user. Hasilnya berupa matriks kemiripan antar item.

3. **Fungsi Rekomendasi Item-based**
   - Untuk menghasilkan rekomendasi, sistem mencari film yang paling mirip dengan film yang sudah diberi rating tinggi oleh user. Skor kemiripan dikalikan dengan rating user pada film tersebut, lalu diurutkan untuk mendapatkan Top-N film yang paling relevan (kecuali film itu sendiri).
   - Fungsi `get_item_based_recommendations` digunakan untuk mengambil rekomendasi film berdasarkan movieId dan rating user.

4. **Contoh Penggunaan**
   - Jika user memberi rating 5 pada film Toy Story (movieId=1), maka sistem akan merekomendasikan 10 film lain yang paling mirip berdasarkan pola rating user lain.

Pendekatan ini efektif untuk menangkap pola preferensi kolektif user terhadap film, namun performanya sangat bergantung pada kelengkapan data rating dan dapat terpengaruh oleh masalah cold-start pada film baru.

#### Contoh Hasil Rekomendasi (Item-based CF)

Sebagai contoh, jika user memberi rating 5 pada film Toy Story (movieId=1), maka sistem merekomendasikan 10 film lain yang paling mirip berdasarkan pola rating user lain:

| Urutan | Judul Film                                      |
|--------|-------------------------------------------------|
| 1      | Star Wars: Episode IV - A New Hope (1977)       |
| 2      | Forrest Gump (1994)                             |
| 3      | Lion King, The (1994)                           |
| 4      | Jurassic Park (1993)                            |
| 5      | Mission: Impossible (1996)                      |
| 6      | Independence Day (a.k.a. ID4) (1996)            |
| 7      | Star Wars: Episode VI - Return of the Jedi (1983)|
| 8      | Groundhog Day (1993)                            |
| 9      | Back to the Future (1985)                       |
| 10     | Toy Story 2 (1999)                              |

Hasil ini menunjukkan bahwa sistem mampu merekomendasikan film-film populer dan relevan yang sering diberi rating tinggi oleh user lain yang juga menyukai Toy Story.

## Evaluation – Content-Based Filtering

#### Metrik Evaluasi yang Digunakan

Untuk mengevaluasi performa sistem rekomendasi berbasis *Content-Based Filtering*, kami menggunakan metrik **Precision**. Precision adalah ukuran seberapa banyak item yang direkomendasikan benar-benar relevan bagi pengguna.

Rumus Precision:

$$
\text{Precision} = \frac{\text{Jumlah item relevan yang direkomendasikan}}{\text{Jumlah total item yang direkomendasikan}}
$$

Dalam konteks proyek ini:

* Item relevan = film yang pernah diberi rating tinggi (≥ 4) oleh user.
* Item direkomendasikan = 10 film hasil prediksi sistem berdasarkan kemiripan konten.

#### Proses Evaluasi

Evaluasi dilakukan terhadap beberapa user yang memiliki data rating. Langkah-langkah evaluasi:

1. Ambil film yang pernah diberi rating tinggi oleh user.
2. Pilih salah satu film sebagai **query**.
3. Dapatkan 10 rekomendasi berdasarkan kemiripan konten dengan film tersebut.
4. Hitung berapa dari 10 rekomendasi tersebut juga termasuk dalam daftar film yang disukai user.
5. Hitung precision untuk setiap user.

Contoh hasil evaluasi:

| userId | Query Film                                 | Precision |
| ------ | ------------------------------------------ | --------- |
| 4      | Nobody Loves Me (Keiner liebt mich) (1994) | 0.10      |
| 100    | American President, The (1995)             | 0.10      |
| 20     | Balto (1995)                               | 0.20      |
| 30     | Braveheart (1995)                          | 0.00      |
| 23     | Heat (1995)                                | 0.00      |
| 50     | Taxi Driver (1976)                         | 0.00      |
| 60     | Shawshank Redemption, The (1994)           | 0.10      |
| 70     | Dead Man Walking (1995)                    | 0.20      |
| 80     | Twelve Monkeys (a.k.a. 12 Monkeys) (1995)  | 0.30      |
| 90     | Sabrina (1995)                             | 0.10      |
| 111    | Casino (1995)                              | 0.10      |

#### Interpretasi Hasil

* Precision untuk sebagian besar user berkisar antara **0.0 hingga 0.3**, artinya dari 10 film yang direkomendasikan, hanya 0 hingga 3 film yang benar-benar disukai user.
* **Nilai precision yang relatif rendah** ini mengindikasikan bahwa sistem rekomendasi belum sepenuhnya mampu menyarankan film yang cocok dengan preferensi pengguna secara personal.
* Hal ini wajar, karena pendekatan *Content-Based Filtering* hanya mempertimbangkan **kemiripan antar item**, bukan preferensi kolektif user lain.

#### Evaluasi terhadap Pendekatan

**Kelebihan**:

* Tidak membutuhkan data dari user lain (cocok untuk new user dengan sedikit data).
* Rekomendasi bersifat personal berdasarkan riwayat minat pengguna sendiri.

**Kekurangan**:

* Cenderung memberikan rekomendasi yang terlalu mirip.
* Kurang efektif untuk user dengan data rating yang sangat terbatas.
* Tidak bisa menangkap tren kolektif seperti *Collaborative Filtering*.


## Evaluation – Collaborative Filtering (RMSE)

Pada pendekatan **Collaborative Filtering (Item-Based)** ini, metrik evaluasi yang digunakan adalah **Root Mean Squared Error (RMSE)**. RMSE merupakan metrik yang umum digunakan untuk mengukur performa model regresi, dengan cara menghitung rata-rata akar kuadrat dari selisih antara nilai aktual dan nilai prediksi. Dalam konteks sistem rekomendasi, RMSE mengukur seberapa dekat prediksi rating terhadap rating sebenarnya yang diberikan pengguna.

#### Hasil Evaluasi Keseluruhan

Data rating dibagi menjadi 80% data latih dan 20% data uji. Model similarity antar item dibangun dari data latih, dan prediksi rating dilakukan untuk pasangan (user, item) pada data uji. Dari hasil evaluasi, diperoleh nilai RMSE sebesar:

```
RMSE Collaborative Filtering (item-based): 0.92
```

Nilai RMSE tersebut menunjukkan bahwa rata-rata prediksi rating memiliki selisih sekitar 0.92 terhadap nilai aktual dalam skala rating 0–5. Ini berarti model cukup baik dalam memprediksi preferensi pengguna terhadap film, meskipun masih terdapat ruang untuk perbaikan.

#### Hasil Evaluasi per Pengguna

Selain evaluasi secara keseluruhan, dilakukan juga evaluasi untuk beberapa pengguna secara individu. RMSE dihitung berdasarkan prediksi dan rating aktual dari masing-masing pengguna. Berikut adalah contoh hasil RMSE untuk 10 pengguna pertama:

| userId | RMSE |
| ------ | ---- |
| 432    | 0.80 |
| 288    | 0.79 |
| 599    | 0.78 |
| 42     | 1.04 |
| 75     | 1.57 |
| 51     | 1.46 |
| 354    | 0.56 |
| 416    | 1.08 |
| 438    | 1.03 |
| 73     | 0.90 |

Hasil ini menunjukkan bahwa performa model dapat bervariasi antar pengguna. Beberapa pengguna memiliki RMSE yang rendah (< 0.8), yang menandakan model cukup akurat dalam memprediksi rating mereka. Namun, terdapat juga pengguna dengan RMSE > 1.0, yang mengindikasikan bahwa prediksi model kurang sesuai dengan preferensi sebenarnya dari pengguna tersebut.

#### Kesimpulan RMSE

Metrik RMSE telah berhasil memberikan gambaran umum mengenai performa model item-based Collaborative Filtering. Secara keseluruhan, model menunjukkan performa yang cukup baik, namun masih perlu peningkatan agar lebih konsisten dalam memberikan prediksi akurat bagi seluruh pengguna.

## Conclusion

Berdasarkan hasil eksplorasi data, pembangunan model, dan evaluasi performa, proyek *MovieLens Recommendation System* ini berhasil membangun dua pendekatan sistem rekomendasi, yaitu **Content-Based Filtering** dan **Collaborative Filtering**, dengan hasil yang cukup memuaskan.

1. **Content-Based Filtering** berhasil menghasilkan rekomendasi film yang relevan dengan preferensi pengguna berdasarkan kemiripan konten, khususnya genre film. Evaluasi menggunakan metrik *precision\@10* menunjukkan bahwa sebagian besar rekomendasi berada dalam kategori yang sesuai dengan film yang disukai pengguna. Pendekatan ini sangat berguna dalam kondisi cold-start untuk pengguna baru yang belum banyak memberi rating, namun sudah menunjukkan ketertarikan terhadap genre tertentu.

2. **Collaborative Filtering** menggunakan pendekatan Matrix Factorization (*SVD*) untuk mempelajari pola rating antar pengguna dan film. Model ini mampu memprediksi rating dengan cukup baik, yang ditunjukkan oleh nilai *Root Mean Squared Error (RMSE)* yang rendah pada data uji. Pendekatan ini lebih efektif dalam mengenali selera pengguna dari interaksi kolektif di antara pengguna lainnya, namun memiliki kelemahan jika data rating sangat spars (jarang diisi).

3. Hasil *exploratory data analysis* (EDA) mengungkapkan bahwa terdapat ketimpangan distribusi rating, baik dari sisi jumlah rating per pengguna maupun jumlah rating per film. Beberapa film populer sangat mendominasi jumlah rating, sementara sebagian besar film hanya memiliki sedikit rating. Hal ini penting untuk diperhatikan dalam pengembangan sistem rekomendasi, agar rekomendasi tidak terlalu bias terhadap film populer saja.

4. Secara keseluruhan, kedua pendekatan memiliki keunggulan masing-masing, dan akan lebih efektif jika digabungkan dalam bentuk sistem rekomendasi hybrid untuk memberikan hasil yang lebih akurat dan personal.

Melalui proyek ini, sistem rekomendasi yang dibangun mampu memberikan kontribusi nyata dalam membantu pengguna menemukan film yang sesuai dengan preferensinya, sekaligus memberikan insight berharga bagi pengelola platform dalam memahami pola konsumsi dan preferensi pengguna. Ke depan, sistem dapat dikembangkan lebih lanjut dengan memasukkan fitur tambahan seperti tag, metadata dari IMDb atau TMDb, serta penggunaan model deep learning untuk meningkatkan akurasi dan fleksibilitas sistem rekomendasi.