# 🍽️ PlatGizi – Smart Menu Planner MBG

> Sistem Rekomendasi Menu Bergizi Harian berbasis Machine Learning untuk mendukung Program **Makan Bergizi Gratis (MBG)** — SDG 2 & SDG 3

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://platgizi.streamlit.app)

---

## 📌 Deskripsi Project

**PlatGizi** adalah aplikasi web interaktif yang merekomendasikan menu makanan harian bergizi seimbang menggunakan algoritma Machine Learning. Sistem ini dirancang khusus untuk mendukung program **Makan Bergizi Gratis (MBG)** pemerintah Indonesia dengan mempertimbangkan kebutuhan kalori dan gizi berbagai kelompok pengguna.

### 🎯 SDGs yang Didukung
- **SDG 2** – Zero Hunger (Tanpa Kelaparan)
- **SDG 3** – Good Health and Well-being (Kesehatan yang Baik)

---

## 🤖 Algoritma Machine Learning

| Tahap | Algoritma | Fungsi |
|-------|-----------|--------|
| Clustering | **K-Means** (K=4) | Mengelompokkan makanan berdasarkan profil gizi |
| Rekomendasi | **Content-Based Filtering** | Merekomendasikan menu sesuai target gizi |
| Similarity | **Cosine Similarity** | Mengukur kemiripan profil gizi makanan |
| Optimasi K | **Elbow Method + Silhouette Score** | Menentukan jumlah cluster optimal |

### Hasil Clustering
| Cluster | Karakteristik | Jumlah Makanan |
|---------|--------------|----------------|
| 🟢 Cluster 0 | Rendah Kalori (Sayuran & Buah) | 811 |
| 🔴 Cluster 1 | Tinggi Kalori & Lemak | 68 |
| 🟡 Cluster 2 | Tinggi Karbo (Nasi & Umbi) | 352 |
| 🔵 Cluster 3 | Tinggi Protein (Lauk Pauk) | 115 |

---

## 📦 Dataset

| Dataset | Sumber | Jumlah Data |
|---------|--------|-------------|
| Kandungan Gizi Makanan Indonesia | [Kaggle](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset) | 1.346 makanan |
| Resep Masakan Indonesia (8 kategori) | [Kaggle](https://www.kaggle.com/datasets/canggih/indonesian-food-recipes) | ~12.000 resep |

---

## 👤 Profil Pengguna & Target Gizi

| Profil | Kalori | Protein | Lemak | Karbo |
|--------|--------|---------|-------|-------|
| 🧒 Anak SD Kelas 1–3 | 1.400 kkal | 35g | 40g | 220g |
| 👦 Anak SD Kelas 4–6 | 1.600 kkal | 40g | 45g | 250g |
| 🎓 Siswa SMP/SMA | 1.800 kkal | 50g | 50g | 280g |
| 🤱 Ibu Hamil/Menyusui | 2.100 kkal | 60g | 60g | 320g |

*Sumber: Angka Kecukupan Gizi (AKG) Kemenkes RI 2019*

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone repository
```bash
git clone https://github.com/username/platgizi.git
cd platgizi
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi
```bash
streamlit run app.py
```

### 4. Buka browser
```
http://localhost:8501
```

---

## 📁 Struktur Project

```
platgizi/
├── app.py                  # Aplikasi Streamlit utama
├── recommender.pkl         # Model ML yang sudah ditraining
├── requirements.txt        # Daftar dependencies
├── .streamlit/
│   └── config.toml         # Konfigurasi tema Streamlit
└── README.md
```

> **Catatan:** File `recommender.pkl` berisi model K-Means, scaler, dan dataset yang sudah diproses. File ini di-generate dari notebook `PlatGizi.ipynb` di Google Colab.

---

## 🛠️ Tech Stack

- **Python** 3.12
- **Streamlit** — Framework aplikasi web
- **Scikit-learn** — K-Means Clustering & Cosine Similarity
- **Pandas & NumPy** — Pengolahan data
- **Font Awesome** — Icon library
- **Google Fonts** — Typography (Nunito + Playfair Display)

---

## 📊 Alur Sistem

```
Input User (Profil + Jumlah Hari)
        ↓
Load recommender.pkl
        ↓
Tentukan target kalori/gizi harian
        ↓
Cosine Similarity → cari makanan per cluster
        ↓
Filter kalori agar sesuai target
        ↓
Generate menu harian (Sarapan + Siang + Malam)
        ↓
Tampilkan menu + progress bar gizi
```

---

## 👥 Tim

**Machine Learning — Binus University LC01**

| No | Nama | NIM |
|----|------|-----|
| 1  | Sabrina Arfanindia D     |  2802448755   |
| 2  | HERLINDA ANGELICA TANJAYA     | 2802397754    |
| 3  |  KRISTIAN NOVAN    |  2802458560   |

---

## 📄 Lisensi

Project ini dibuat untuk keperluan tugas kuliah Machine Learning — Binus University.
