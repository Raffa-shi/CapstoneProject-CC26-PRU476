import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import string
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from collections import Counter
from utils.preprocessor import full_preprocessing
import warnings
warnings.filterwarnings('ignore')


# CONFIG

st.set_page_config(
    page_title="BPJS Sentiment Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# THEME & CSS


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --bpjs-green:    #00a651;
    --bpjs-green-dk: #007a3d;
    --bpjs-green-lt: #e6f7ee;
    --bpjs-blue:     #003d7c;
    --bpjs-blue-lt:  #e8f0fb;
    --accent-gold:   #f5a623;
    --pos-color:     #00a651;
    --net-color:     #f5a623;
    --neg-color:     #e63946;
    --bg-main:       #f4f7f5;
    --bg-card:       #ffffff;
    --text-main:     #1a2e1e;
    --text-muted:    #5a7261;
    --border:        #d4e8db;
    --shadow:        0 2px 16px rgba(0,166,81,0.10);
    --shadow-hover:  0 8px 32px rgba(0,166,81,0.18);
    --radius:        14px;
    --radius-lg:     20px;
}

* { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp { background: var(--bg-main); }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1280px !important; }

/* ---- HEADER ---- */
.hero-header {
    background: linear-gradient(135deg, var(--bpjs-blue) 0%, var(--bpjs-green-dk) 60%, var(--bpjs-green) 100%);
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    padding: 2.5rem 3rem 2rem 3rem;
    margin: 0 -2rem 2rem -2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,166,81,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: rgba(255,255,255,0.80);
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
}
.hero-stat {
    text-align: center;
}
.hero-stat-val {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'Space Grotesk', monospace;
}
.hero-stat-lbl {
    color: rgba(255,255,255,0.70);
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* ---- SECTION TITLE ---- */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--bpjs-blue);
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 4px; height: 20px;
    background: var(--bpjs-green);
    border-radius: 2px;
}

/* ---- CARDS ---- */
.card {
    background: var(--bg-card);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    padding: 1.4rem 1.6rem;
    transition: box-shadow 0.2s, transform 0.2s;
    height: 100%;
}
.card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.card-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--bpjs-blue);
    font-family: 'Space Grotesk', monospace;
    line-height: 1;
}
.card-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 4px;
}
.card-icon {
    font-size: 1.6rem;
    margin-bottom: 8px;
}

/* ---- TECH STACK CARDS ---- */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1rem;
}
.tech-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    text-align: center;
    transition: all 0.2s;
    cursor: default;
}
.tech-card:hover {
    border-color: var(--bpjs-green);
    box-shadow: 0 4px 16px rgba(0,166,81,0.12);
    transform: translateY(-3px);
}
.tech-icon { font-size: 1.8rem; margin-bottom: 6px; }
.tech-name {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--bpjs-blue);
}
.tech-desc {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 3px;
    line-height: 1.3;
}

/* ---- PREDICTION RESULT ---- */
.pred-box {
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    border: 2px solid;
    transition: all 0.3s;
}
.pred-box.positif {
    background: linear-gradient(135deg, #e6f7ee 0%, #c8f0d8 100%);
    border-color: var(--pos-color);
}
.pred-box.netral {
    background: linear-gradient(135deg, #fff8e6 0%, #fef3cc 100%);
    border-color: var(--net-color);
}
.pred-box.negatif {
    background: linear-gradient(135deg, #fdeaea 0%, #fccfcf 100%);
    border-color: var(--neg-color);
}
.pred-emoji { font-size: 3.5rem; }
.pred-label {
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 8px;
    font-family: 'Space Grotesk', sans-serif;
}
.pred-label.positif { color: var(--pos-color); }
.pred-label.netral  { color: #b07d00; }
.pred-label.negatif { color: var(--neg-color); }
.pred-conf {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ---- CONFIDENCE BAR ---- */
.conf-bar-wrap { margin: 6px 0; }
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 3px;
}
.conf-bar-bg {
    background: #e8f0eb;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ---- INSIGHT CHIP ---- */
.insight-chip {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px;
}
.chip-positif { background: #e6f7ee; color: var(--pos-color); border: 1px solid #a8dfc0; }
.chip-netral  { background: #fff8e6; color: #9a6c00; border: 1px solid #f5d78e; }
.chip-negatif { background: #fdeaea; color: var(--neg-color); border: 1px solid #f5b8b8; }

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bpjs-green) !important;
    color: white !important;
}

/* ---- INPUT AREA ---- */
.stTextArea textarea {
    border: 2px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: var(--bpjs-green) !important;
    box-shadow: 0 0 0 3px rgba(0,166,81,0.1) !important;
}

/* ---- BUTTON ---- */
.stButton > button {
    background: linear-gradient(135deg, var(--bpjs-green-dk), var(--bpjs-green)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(0,166,81,0.25) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(0,166,81,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ---- METRIC CARDS ---- */
.metric-strip {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.metric-pill {
    background: var(--bpjs-green-lt);
    border: 1px solid #b8e8cc;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--bpjs-green-dk);
    font-family: 'Space Grotesk', monospace;
}
.metric-pill span { color: var(--text-muted); font-weight: 400; }

/* ---- DIVIDER ---- */
.custom-divider {
    border: none;
    border-top: 1.5px solid var(--border);
    margin: 1.5rem 0;
}

/* ---- SCROLLABLE TABLE ---- */
.recent-table { overflow-x: auto; }

/* ---- EXAMPLE CHIPS ---- */
.example-row { display: flex; flex-wrap: wrap; gap: 8px; }
.ex-chip {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-main);
    cursor: pointer;
    transition: all 0.15s;
}
.ex-chip:hover { border-color: var(--bpjs-green); color: var(--bpjs-green); }
</style>
""", unsafe_allow_html=True)


# LOAD RESOURCES


@st.cache_resource
def load_model():
    model     = joblib.load('model/sentiment_model.pkl')
    vectorizer= joblib.load('model/tfidf_vectorizer.pkl')
    encoder   = joblib.load('model/label_encoder.pkl')
    return model, vectorizer, encoder

@st.cache_data
def load_data():
    df = pd.read_csv('data/bpjs_labeled_dataset.csv')
    with open('data/wordcloud_data.json', 'r', encoding='utf-8') as f:
        wc_data = json.load(f)
    with open('data/model_stats.json', 'r', encoding='utf-8') as f:
        stats = json.load(f)
    return df, wc_data, stats

@st.cache_resource
def load_preprocessor():
    from utils.preprocessor import full_preprocessing
    return full_preprocessing
try:
    model, vectorizer, encoder = load_model()
    df, wc_data, stats         = load_data()
    full_preprocessing         = load_preprocessor()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    load_error   = str(e)


# SESSION STATE


if 'history' not in st.session_state:
    st.session_state.history = []
if 'input_text' not in st.session_state:
    st.session_state.input_text = ''


# HELPER FUNCTIONS


def predict_sentiment(teks_raw):
    """Prediksi sentimen dan return label + probabilitas semua kelas."""
    processed = full_preprocessing(teks_raw)
    if not processed.strip():
        return None, None, None, None
    vector    = vectorizer.transform([processed])
    pred_enc  = model.predict(vector)[0]
    proba     = model.predict_proba(vector)[0]
    label     = encoder.inverse_transform([pred_enc])[0]
    proba_dict= dict(zip(encoder.classes_, proba))
    confidence= max(proba)
    return label, confidence, proba_dict, processed

def get_sentiment_emoji(label):
    return {'Positif': '😊', 'Netral': '😐', 'Negatif': '😟'}.get(label, '❓')

def get_sentiment_css(label):
    return label.lower()

def make_wordcloud(text, colormap, bg='white'):
    if not text or not text.strip():
        return None
    wc = WordCloud(
        width=800, height=350,
        background_color=bg,
        colormap=colormap,
        max_words=80,
        collocations=False,
        random_state=42,
        min_font_size=10,
        max_font_size=90,
        prefer_horizontal=0.85
    ).generate(text)
    return wc

def render_conf_bar(label, value, color):
    st.markdown(f"""
    <div class="conf-bar-wrap">
        <div class="conf-bar-label">
            <span>{label}</span><span>{value*100:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{value*100:.1f}%;background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# HERO HEADER


total_data   = stats.get('total_data', 2432) if MODEL_LOADED else 2432
label_dist   = stats.get('label_dist', {})   if MODEL_LOADED else {}
model_acc    = stats.get('accuracy', 0.75)    if MODEL_LOADED else 0
model_f1     = stats.get('f1_macro', 0.73)    if MODEL_LOADED else 0

neg_pct = stats.get('label_pct', {}).get('Negatif', 41) if MODEL_LOADED else 41
pos_pct = stats.get('label_pct', {}).get('Positif', 4)  if MODEL_LOADED else 4
net_pct = stats.get('label_pct', {}).get('Netral', 54)  if MODEL_LOADED else 54

st.markdown(f"""
<div class="hero-header">
    <div class="hero-badge">🏥 BPJS Kesehatan · NLP Sentiment Analysis</div>
    <h1 class="hero-title">Dashboard Analisis Sentimen<br>Komentar Instagram BPJS</h1>
    <p class="hero-subtitle">
        Klasifikasi sentimen real-time menggunakan TF-IDF + SVM · Dataset Instagram BPJS Kesehatan
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-val">{total_data:,}</div>
            <div class="hero-stat-lbl">Total Komentar</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">{model_acc*100:.1f}%</div>
            <div class="hero-stat-lbl">Akurasi Model</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">{model_f1:.3f}</div>
            <div class="hero-stat-lbl">F1-Macro</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">3</div>
            <div class="hero-stat-lbl">Kelas Sentimen</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# TABS NAVIGASI

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediksi Sentimen",
    "📊 Analitik Dataset",
    "☁️ Word Cloud",
    "⚙️ Tech Stack & Model"
])


# TAB 1 — PREDIKSI SENTIMEN


with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ---- INPUT PANEL ----
    with col_left:
        st.markdown('<p class="section-title">Input Komentar</p>', unsafe_allow_html=True)

        # Contoh komentar
        st.markdown("""
        <p style="font-size:0.78rem;color:var(--text-muted);margin-bottom:8px;font-weight:500;">
        Contoh komentar · klik untuk isi otomatis:
        </p>
        """, unsafe_allow_html=True)

        examples = {
            "😊 Positif": [
                "Alhamdulillah sudah aktif kembali, terima kasih BPJS!",
                "Pelayanan sangat membantu dan cepat.",
            ],
            "😐 Netral": [
                "Mau tanya cara daftar BPJS mandiri bagaimana?",
                "Apakah bisa pindah faskes secara online?",
            ],
            "😟 Negatif": [
                "Aplikasi error terus tidak bisa login dari kemarin.",
                "Percuma bayar iuran tapi ditolak saat berobat!",
            ],
        }

        ex_cols = st.columns(3)
        for ci, (cat, exs) in enumerate(examples.items()):
            with ex_cols[ci]:
                for ex in exs:
                    if st.button(ex[:35] + "…", key=f"ex_{ex[:10]}", use_container_width=True):
                        st.session_state.input_text = ex

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        user_input = st.text_area(
            "Masukkan teks komentar:",
            value=st.session_state.input_text,
            height=130,
            placeholder="Ketik atau tempel komentar Instagram BPJS di sini...",
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            analyze_btn = st.button("🔍 Analisis Sentimen", type="primary")
        with col_btn2:
            clear_btn = st.button("🗑 Hapus", type="secondary")

        if clear_btn:
            st.session_state.input_text = ''
            st.rerun()

        # Statistik input
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            st.markdown(f"""
            <div class="metric-strip" style="margin-top:8px;">
                <div class="metric-pill"><span>Kata: </span>{word_count}</div>
                <div class="metric-pill"><span>Karakter: </span>{char_count}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- OUTPUT PANEL ----
    with col_right:
        st.markdown('<p class="section-title">Hasil Prediksi</p>', unsafe_allow_html=True)

        if analyze_btn and user_input.strip():
            if not MODEL_LOADED:
                st.error(f"Model belum dimuat. Jalankan cell serialisasi dulu.\n{load_error}")
            else:
                with st.spinner("Menganalisis..."):
                    label, confidence, proba_dict, processed = predict_sentiment(user_input)

                if label is None:
                    st.warning("Teks terlalu pendek atau kosong setelah preprocessing.")
                else:
                    css_cls = get_sentiment_css(label)
                    emoji   = get_sentiment_emoji(label)

                    st.markdown(f"""
                    <div class="pred-box {css_cls}">
                        <div class="pred-emoji">{emoji}</div>
                        <div class="pred-label {css_cls}">{label}</div>
                        <div class="pred-conf">Confidence: {confidence*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                    # Confidence bars semua kelas
                    st.markdown("**Distribusi Probabilitas:**")
                    color_map = {'Positif': '#00a651', 'Netral': '#f5a623', 'Negatif': '#e63946'}
                    for lbl in ['Positif', 'Netral', 'Negatif']:
                        val = proba_dict.get(lbl, 0)
                        render_conf_bar(lbl, val, color_map[lbl])

                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                    # Processed text
                    with st.expander("Lihat teks setelah preprocessing"):
                        st.code(processed if processed else "(kosong)", language=None)

                    # Simpan ke history
                    st.session_state.history.insert(0, {
                        'teks'      : user_input[:80] + ('…' if len(user_input) > 80 else ''),
                        'label'     : label,
                        'conf'      : f"{confidence*100:.1f}%",
                    })
                    if len(st.session_state.history) > 20:
                        st.session_state.history.pop()

        elif not analyze_btn:
            st.markdown("""
            <div style="
                border: 2px dashed var(--border);
                border-radius: var(--radius-lg);
                padding: 3rem 2rem;
                text-align: center;
                color: var(--text-muted);
            ">
                <div style="font-size:2.5rem;margin-bottom:12px;">🔍</div>
                <div style="font-size:0.9rem;font-weight:500;">
                    Masukkan komentar dan klik<br><strong>Analisis Sentimen</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- RIWAYAT ----
    if st.session_state.history:
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Riwayat Prediksi Sesi Ini</p>', unsafe_allow_html=True)

        rows = []
        for h in st.session_state.history[:10]:
            badge = {'Positif': '🟢', 'Netral': '🟡', 'Negatif': '🔴'}.get(h['label'], '⚪')
            rows.append({
                'Sentimen'  : f"{badge} {h['label']}",
                'Confidence': h['conf'],
                'Komentar'  : h['teks'],
            })
        hist_df = pd.DataFrame(rows)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if st.button("🗑 Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()


# TAB 2 — ANALITIK DATASET


with tab2:
    st.markdown('<p class="section-title">Ringkasan Dataset</p>', unsafe_allow_html=True)

    # Metrik utama
    m1, m2, m3, m4 = st.columns(4)

    metrik_items = [
        ("Total Komentar", f"{total_data:,}", "Setelah cleaning", "💬"),
        ("Komentar Negatif", f"{label_dist.get('Negatif', 999):,}", f"{neg_pct}% dari total", "😟"),
        ("Komentar Netral",  f"{label_dist.get('Netral', 1325):,}", f"{net_pct}% dari total", "😐"),
        ("Komentar Positif", f"{label_dist.get('Positif', 108):,}", f"{pos_pct}% dari total", "😊"),
    ]
    for col, (label, val, sub, icon) in zip([m1, m2, m3, m4], metrik_items):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-icon">{icon}</div>
                <div class="card-label">{label}</div>
                <div class="card-value">{val}</div>
                <div class="card-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Distribusi & Pie
    col_pie, col_bar = st.columns([1, 1], gap="large")

    with col_pie:
        st.markdown('<p class="section-title">Distribusi Sentimen</p>', unsafe_allow_html=True)

        if MODEL_LOADED and label_dist:
            labels_list = list(label_dist.keys())
            sizes       = [label_dist[l] for l in labels_list]
            colors_pie  = {'Negatif': '#e63946', 'Netral': '#f5a623', 'Positif': '#00a651'}
            clrs        = [colors_pie.get(l, '#888') for l in labels_list]

            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            fig_pie.patch.set_alpha(0)
            wedges, texts, autotexts = ax_pie.pie(
                sizes,
                labels=labels_list,
                autopct='%1.1f%%',
                colors=clrs,
                startangle=90,
                explode=[0.04] * len(labels_list),
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontsize': 10, 'fontweight': '600'}
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_color('white')
                at.set_fontweight('bold')
            ax_pie.set_title('Proporsi Kelas Sentimen', fontsize=11,
                             fontweight='bold', color='#1a2e1e', pad=10)
            st.pyplot(fig_pie, use_container_width=True)
            plt.close()

    with col_bar:
        st.markdown('<p class="section-title">Jumlah per Kelas</p>', unsafe_allow_html=True)

        if MODEL_LOADED and label_dist:
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            fig_bar.patch.set_alpha(0)

            ordered    = ['Negatif', 'Netral', 'Positif']
            bar_vals   = [label_dist.get(l, 0) for l in ordered]
            bar_colors = ['#e63946', '#f5a623', '#00a651']

            bars = ax_bar.bar(ordered, bar_vals, color=bar_colors,
                              width=0.5, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, bar_vals):
                ax_bar.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 10, str(val),
                            ha='center', fontsize=10, fontweight='bold', color='#1a2e1e')

            ax_bar.set_ylabel('Jumlah Komentar', fontsize=9)
            ax_bar.set_title('Jumlah Komentar per Sentimen', fontsize=11,
                             fontweight='bold', color='#1a2e1e')
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.grid(axis='y', alpha=0.25)
            ax_bar.set_facecolor('none')
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Top keywords per sentimen
    st.markdown('<p class="section-title">Top 15 Kata per Sentimen</p>', unsafe_allow_html=True)

    kw_cols = st.columns(3)
    kw_cfg  = [
        ('Positif', '#00a651', 'Greens_r'),
        ('Netral',  '#f5a623', 'YlOrBr_r'),
        ('Negatif', '#e63946', 'Reds_r'),
    ]

    if MODEL_LOADED:
        for col, (sent, color, cmap) in zip(kw_cols, kw_cfg):
            with col:
                text_sent = wc_data.get(sent, '')
                if text_sent:
                    words   = text_sent.split()
                    counter = Counter(words)
                    top15   = counter.most_common(15)

                    if top15:
                        words_top, counts_top = zip(*top15)
                        fig_kw, ax_kw = plt.subplots(figsize=(4, 5))
                        fig_kw.patch.set_alpha(0)
                        colors_kw = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(top15)))
                        ax_kw.barh(list(words_top)[::-1], list(counts_top)[::-1],
                                   color=colors_kw, edgecolor='white')
                        ax_kw.set_title(f'{sent}', fontsize=10, fontweight='bold', color=color)
                        ax_kw.set_xlabel('Frekuensi', fontsize=8)
                        ax_kw.spines['top'].set_visible(False)
                        ax_kw.spines['right'].set_visible(False)
                        ax_kw.grid(axis='x', alpha=0.2)
                        ax_kw.set_facecolor('none')
                        ax_kw.tick_params(labelsize=8)
                        plt.tight_layout()
                        st.pyplot(fig_kw, use_container_width=True)
                        plt.close()

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Sample dataset
    st.markdown('<p class="section-title">Sample Dataset Berlabel</p>', unsafe_allow_html=True)

    if MODEL_LOADED:
        filter_sent = st.selectbox(
            "Filter sentimen:",
            ['Semua', 'Positif', 'Netral', 'Negatif'],
            label_visibility='collapsed'
        )
        df_show = df if filter_sent == 'Semua' else df[df['sentimen'] == filter_sent]
        st.dataframe(
            df_show[['Komentar', 'sentimen']].sample(min(20, len(df_show)), random_state=42),
            use_container_width=True,
            hide_index=True,
            height=320
        )


# TAB 3 — WORD CLOUD


with tab3:
    st.markdown('<p class="section-title">Word Cloud per Kelas Sentimen</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:1.2rem;">
    Ukuran kata merepresentasikan frekuensi kemunculan dalam dataset setelah preprocessing.
    Setiap kelas ditampilkan terpisah untuk analisis yang lebih fokus.
    </p>
    """, unsafe_allow_html=True)

    wc_cfg = [
        ('Positif', 'Greens',   '#e6f7ee', '#007a3d'),
        ('Netral',  'YlOrBr',   '#fff8e6', '#9a6c00'),
        ('Negatif', 'Reds',     '#fdeaea', '#c0392b'),
    ]

    for sent, cmap, bg_color, title_color in wc_cfg:
        st.markdown(f"""
        <div style="
            background:{bg_color};
            border-radius:var(--radius-lg);
            padding:1.2rem 1.6rem 0.5rem;
            margin-bottom:1.2rem;
        ">
        <p style="font-size:1rem;font-weight:700;color:{title_color};margin:0 0 8px 0;">
            {get_sentiment_emoji({'Positif':'😊','Netral':'😐','Negatif':'😟'}[sent])} 
            Sentimen {sent} 
            <span style="font-size:0.75rem;font-weight:400;color:var(--text-muted);">
            · n={label_dist.get(sent, 0):,} komentar
            </span>
        </p>
        """, unsafe_allow_html=True)

        text_wc = wc_data.get(sent, '') if MODEL_LOADED else ''
        if text_wc and text_wc.strip():
            wc = make_wordcloud(text_wc, cmap)
            if wc:
                fig_wc, ax_wc = plt.subplots(figsize=(12, 4.5))
                fig_wc.patch.set_alpha(0)
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc, use_container_width=True)
                plt.close()
        else:
            st.info("Data tidak tersedia.")

        st.markdown("</div>", unsafe_allow_html=True)

    # WordCloud Custom
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Word Cloud Kustom</p>', unsafe_allow_html=True)

    custom_text = st.text_area(
        "Masukkan teks untuk word cloud:",
        height=100,
        placeholder="Paste teks bebas di sini untuk generate word cloud kustom...",
        label_visibility='collapsed'
    )

    wc_col1, wc_col2 = st.columns([3, 1])
    with wc_col2:
        wc_cmap = st.selectbox("Warna:", ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges'])
    with wc_col1:
        gen_wc_btn = st.button("☁️ Generate Word Cloud", use_container_width=True)

    if gen_wc_btn and custom_text.strip():
        processed_wc = full_preprocessing(custom_text) if MODEL_LOADED else custom_text
        wc_custom    = make_wordcloud(processed_wc or custom_text, wc_cmap)
        if wc_custom:
            fig_cust, ax_cust = plt.subplots(figsize=(12, 4.5))
            ax_cust.imshow(wc_custom, interpolation='bilinear')
            ax_cust.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_cust, use_container_width=True)
            plt.close()

# TAB 4 — TECH STACK & MODEL INFO

with tab4:
    # Tech Stack
    st.markdown('<p class="section-title">Tech Stack</p>', unsafe_allow_html=True)

    tech_items = [
        ("🐍", "Python 3.10", "Bahasa pemrograman utama untuk seluruh pipeline"),
        ("📓", "Jupyter Notebook", "Environment pengembangan dan eksplorasi data"),
        ("🐼", "Pandas", "Manipulasi dan analisis data tabular"),
        ("🔢", "NumPy", "Komputasi numerik dan array processing"),
        ("🤖", "Scikit-learn", "Training model, evaluasi, dan pipeline ML"),
        ("📝", "TF-IDF", "Feature extraction teks — unigram + bigram"),
        ("⚡", "SVM Linear", "Model klasifikasi utama dengan class_weight balanced"),
        ("📊", "Naive Bayes", "Model baseline pembanding — MultinomialNB"),
        ("🌏", "PySastrawi", "Stemming dan stopword bahasa Indonesia"),
        ("🔤", "NLTK", "Tokenisasi dan preprocessing NLP"),
        ("☁️", "WordCloud", "Visualisasi frekuensi kata per kelas sentimen"),
        ("📈", "Matplotlib", "Visualisasi chart dan plot evaluasi model"),
        ("🎨", "Seaborn", "Visualisasi statistik dan confusion matrix"),
        ("💾", "Joblib", "Serialisasi model ke format .pkl"),
        ("🌊", "Streamlit", "Framework deployment dashboard interaktif"),
        ("🔍", "Regex", "Pattern matching untuk preprocessing dan labeling"),
    ]

    cols_per_row = 4
    for row_start in range(0, len(tech_items), cols_per_row):
        row_items = tech_items[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (icon, name, desc) in zip(cols, row_items):
            with col:
                st.markdown(f"""
                <div class="tech-card">
                    <div class="tech-icon">{icon}</div>
                    <div class="tech-name">{name}</div>
                    <div class="tech-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Model Performance
    st.markdown('<p class="section-title">Performa Model</p>', unsafe_allow_html=True)

    perf_cols = st.columns(2)

    with perf_cols[0]:
        st.markdown("""
        <div class="card">
        <div class="card-label">Model Terpilih</div>
        <div style="font-size:1.3rem;font-weight:800;color:var(--bpjs-blue);margin:8px 0;">
            TF-IDF + SVM Linear
        </div>
        """, unsafe_allow_html=True)

        model_metrics = [
            ("Accuracy",     stats.get('accuracy',   0.75) if MODEL_LOADED else 0.75),
            ("Precision",    stats.get('precision',  0.75) if MODEL_LOADED else 0.75),
            ("Recall",       stats.get('recall',     0.75) if MODEL_LOADED else 0.75),
            ("F1-Weighted",  stats.get('f1_weighted',0.75) if MODEL_LOADED else 0.75),
            ("F1-Macro",     stats.get('f1_macro',   0.73) if MODEL_LOADED else 0.73),
        ]
        for met_name, met_val in model_metrics:
            bar_color = '#00a651' if met_val >= 0.70 else ('#f5a623' if met_val >= 0.55 else '#e63946')
            render_conf_bar(met_name, met_val, bar_color)

        st.markdown("</div>", unsafe_allow_html=True)

    with perf_cols[1]:
        vocab  = stats.get('vocab_size',    3000) if MODEL_LOADED else 3000
        tr_sz  = stats.get('train_size',    1945) if MODEL_LOADED else 1945
        te_sz  = stats.get('test_size',      487) if MODEL_LOADED else 487
        ngram  = stats.get('ngram_range', '(1,2)') if MODEL_LOADED else '(1,2)'
        best_c = stats.get('best_C', 1.0) if MODEL_LOADED else 1.0

        config_items = [
            ("Hyperparameter C",       str(best_c)),
            ("Kernel SVM",             "Linear"),
            ("Class Weight",           "Balanced"),
            ("TF-IDF Vocab Size",      f"{vocab:,} fitur"),
            ("N-gram Range",           ngram),
            ("Training Samples",       f"{tr_sz:,}"),
            ("Testing Samples",        f"{te_sz:,}"),
            ("Train-Test Split",       "80% / 20%"),
            ("CV Strategy",            "Stratified K-Fold (k=5)"),
            ("Optimization Metric",    "F1-Macro"),
        ]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Konfigurasi Model</div>', unsafe_allow_html=True)
        for cfg_k, cfg_v in config_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:6px 0;border-bottom:1px solid var(--border);
                        font-size:0.82rem;">
                <span style="color:var(--text-muted);font-weight:500;">{cfg_k}</span>
                <span style="color:var(--bpjs-blue);font-weight:700;
                             font-family:'Space Grotesk',monospace;">{cfg_v}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Pipeline Flow
    st.markdown('<p class="section-title">Alur Pipeline ML</p>', unsafe_allow_html=True)

    pipeline_steps = [
        ("01", "Data Collection",   "Scraping komentar Instagram BPJS", "💬"),
        ("02", "Data Cleaning",     "Hapus duplikat, akun resmi, artifact scraping", "🧹"),
        ("03", "Preprocessing",     "Case fold → URL → emoji → normalisasi → stemming", "⚙️"),
        ("04", "Data Labeling",     "7-layer rule-based: whitelist, hard neg, sarkasme, negasi, scoring", "🏷️"),
        ("05", "EDA",               "Distribusi, wordcloud, n-gram, frekuensi kata", "📊"),
        ("06", "Feature Extraction","TF-IDF Vectorizer (1,2)-gram, 3000 fitur, sublinear TF", "🔢"),
        ("07", "GridSearchCV",      "Cari hyperparameter optimal via Stratified K-Fold CV=5", "🔍"),
        ("08", "Model Training",    "SVM Linear C=best_C, class_weight=balanced", "🤖"),
        ("09", "Evaluation",        "Accuracy, F1-Macro, F1-Weighted, Confusion Matrix, Learning Curve", "📈"),
        ("10", "Serialization",     "Simpan model.pkl, vectorizer.pkl, encoder.pkl via Joblib", "💾"),
        ("11", "Deployment",        "Streamlit dashboard interaktif dengan prediksi real-time", "🚀"),
    ]

    for step_num, step_name, step_desc, step_icon in pipeline_steps:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:16px;
                    padding:12px 0;border-bottom:1px solid var(--border);">
            <div style="
                background:var(--bpjs-green);
                color:white;
                font-size:0.7rem;font-weight:800;
                font-family:'Space Grotesk',monospace;
                width:32px;height:32px;
                border-radius:8px;
                display:flex;align-items:center;justify-content:center;
                flex-shrink:0;
            ">{step_num}</div>
            <div style="flex:1;">
                <div style="font-size:0.88rem;font-weight:700;color:var(--bpjs-blue);">
                    {step_icon} {step_name}
                </div>
                <div style="font-size:0.78rem;color:var(--text-muted);margin-top:2px;">
                    {step_desc}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Insight bisnis
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Insight Bisnis</p>', unsafe_allow_html=True)

    ins_cols = st.columns(3)
    insights = [
        (
            "😟", "Sentimen Dominan",
            f"Negatif {neg_pct}%",
            "Mayoritas komentar adalah keluhan — menunjukkan gap signifikan antara ekspektasi peserta dan kualitas layanan aktual BPJS.",
            "#fdeaea", "#c0392b"
        ),
        (
            "📋", "Isu Utama Publik",
            "Administrasi & Teknis",
            "Top keluhan: aplikasi error, BPJS nonaktif, ditolak RS, tagihan tidak sesuai. Dua cluster berbeda yang butuh respons berbeda.",
            "#fff8e6", "#9a6c00"
        ),
        (
            "💡", "Rekomendasi",
            "Monitoring Berkelanjutan",
            "Model ini dapat diintegrasikan ke social media monitoring untuk deteksi keluhan viral secara otomatis sebelum menjadi krisis reputasi.",
            "#e6f7ee", "#007a3d"
        ),
    ]
    for col, (icon, title, subtitle, desc, bg, color) in zip(ins_cols, insights):
        with col:
            st.markdown(f"""
            <div class="card" style="background:{bg};border-color:{color}33;">
                <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.5px;color:{color};margin-bottom:4px;">{title}</div>
                <div style="font-size:1rem;font-weight:800;color:{color};margin-bottom:8px;">{subtitle}</div>
                <div style="font-size:0.78rem;color:var(--text-muted);line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# FOOTER

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1rem 0;color:var(--text-muted);font-size:0.78rem;">
    <strong style="color:var(--bpjs-green);">BPJS Sentiment Analyzer</strong>
    · Capstone Project · NLP & Machine Learning
    · TF-IDF + SVM · Bahasa Indonesia
</div>
""", unsafe_allow_html=True)