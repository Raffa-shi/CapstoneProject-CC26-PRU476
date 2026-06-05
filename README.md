# CapstoneProject-CC26-PRU476

## Analisis Sentimen Komentar Masyarakat Terkait Postingan "Beberapa Fakta Penting Tentang BPJS" Pada Akun Instagram BPJS Kesehatan

---

<img width="1983" height="793" alt="Cover" src="https://github.com/user-attachments/assets/84af821a-25df-4e81-9764-2f00e753dc25" />

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F6B?style=for-the-badge)
![Sastrawi](https://img.shields.io/badge/Sastrawi-009688?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![WordCloud](https://img.shields.io/badge/WordCloud-2E8B57?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)

---

## Project Overview

Media sosial merupakan salah satu sarana utama masyarakat dalam menyampaikan opini, kritik, saran, maupun apresiasi terhadap layanan publik. Komentar yang diberikan masyarakat sering kali mengandung informasi penting yang dapat digunakan untuk memahami persepsi publik terhadap suatu layanan.

Pada project ini dilakukan analisis sentimen terhadap komentar masyarakat pada postingan Instagram BPJS Kesehatan yang membahas "Beberapa Fakta Penting Tentang BPJS". Tujuan utama project ini adalah membangun model klasifikasi sentimen yang mampu mengelompokkan komentar ke dalam tiga kategori:

- Positif
- Netral
- Negatif

Hasil analisis kemudian diimplementasikan ke dalam dashboard interaktif menggunakan Streamlit sehingga pengguna dapat melakukan prediksi sentimen secara real-time.

---

## Dataset Information

### Sumber Data

Dataset diperoleh melalui proses scraping komentar Instagram pada akun resmi BPJS Kesehatan.

### Jumlah Data

| Keterangan | Jumlah |
|------------|---------|
| Total Komentar | 2.432 |
| Kelas Sentimen | 3 |
| Positif | Label Manual |
| Netral | Label Manual |
| Negatif | Label Manual |

### Struktur Dataset

| Kolom | Deskripsi |
|---------|------------|
| komentar | Komentar asli hasil scraping |
| teks_bersih | Hasil preprocessing |
| sentimen | Label sentimen |

---

## Project Structure

```bash
CapstoneProject-CC26-PRU476/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── bpjs_comments.csv
│   ├── bpjs_clean_dataset.csv
│   ├── bpjs_labeled_dataset.csv
│   └── model_stats.json
│
├── model/
│   ├── svm_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── img/
│   ├── banner.png
│   ├── wordcloud.png
│   └── dashboard_preview.png
│
├── utils/
│   ├── __init__.py
│   └── preprocessor.py
│
├── scrap/
│   └── scraper.py
│
└── notebook/
    └── Analisa_BPJS.ipynb
```

---

## Data Science Pipeline

### 1. Data Collection

Scraping komentar Instagram menggunakan Selenium.

```text
Instagram Post
      
Comment Scraping
      
Raw Dataset
```

---

### 2. Data Understanding

- Mengecek jumlah data
- Missing value
- Duplicate data
- Distribusi komentar

---

### 3. Data Cleaning

- Remove duplicate comments
- Remove empty comments
- Remove noise text

---

### 4. Text Preprocessing

```text
Case Folding
      
Remove URL
      
Remove Mention
      
Remove Hashtag
      
Remove Emoji
      
Remove Number
      
Remove Punctuation
      
Remove Extra Whitespace
      
Normalization
      
Tokenization
      
Stopword Removal
      
Stemming
```

---

### 5. Exploratory Data Analysis

- Distribusi panjang komentar
- Distribusi sentimen
- Top Frequent Words
- Word Frequency Analysis
- Word Cloud Visualization

---

### 6. Data Labeling

Sentimen dikategorikan menjadi:

| Label | Deskripsi |
|---------|------------|
| Positif | Apresiasi dan kepuasan |
| Netral | Pertanyaan atau informasi |
| Negatif | Keluhan dan kritik |

---

### 7. Train Test Split

```python
Train : 80%
Test  : 20%
```

---

### 8. Feature Extraction

#### TF-IDF Vectorization

Mengubah teks menjadi representasi numerik berdasarkan bobot kata.

```text
Komentar
      
TF-IDF
      
Feature Vector
```

---

### 9. Model Development

#### Naive Bayes

Digunakan sebagai baseline model.

#### Support Vector Machine (SVM)

Digunakan sebagai model utama karena menghasilkan performa terbaik.

---

### 10. Model Evaluation

Metrik yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Learning Curve

---

### 11. Deployment

Model yang telah dilatih disimpan menggunakan Joblib dan diintegrasikan ke dalam dashboard Streamlit.

```text
Model (.pkl)
      
Streamlit Dashboard
      
Real-Time Sentiment Prediction
```

---

## Dashboard Features

### Sentiment Prediction

Prediksi sentimen komentar secara real-time.

### Dataset Analytics

Menampilkan statistik dataset hasil scraping.

### Word Cloud

Visualisasi kata dominan pada masing-masing sentimen.

### Model Information

Menampilkan:

- Accuracy
- Precision
- Recall
- F1-Score
- Hyperparameter
- Dataset Information

---

## Model Performance

| Model | Accuracy |
|---------|---------|
| Naive Bayes | 71% |
| SVM Linear | 78% |

SVM dipilih sebagai model utama karena memiliki performa terbaik pada dataset komentar Instagram BPJS Kesehatan.

---

## Installation

Clone repository:

```bash
git clone https://github.com/USERNAME/CapstoneProject-CC26-PRU476.git
cd CapstoneProject-CC26-PRU476
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

---

## Team Members

### CC26-PRU476

| ID | Nama | Role |
|-----|--------|--------|
| CDCC180D6Y1047 | Abdul Hamid Amin | Data Scientist |
| CDCC208D6X1140 | Annisa Ayu Anggraini | Data Scientist |
| CDCC288D6Y1249 | Raafa Syahidul Haq Irsi | Data Scientist |

---

## License

This project was developed for educational and research purposes as part of the Capstone Project Dicoding Program.
