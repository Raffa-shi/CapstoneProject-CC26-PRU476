import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# PAGE CONFIG
st.set_page_config(
    page_title="BPJS Sentiment Analyzer",
    page_icon="https://www.bpjs-kesehatan.go.id/bpjs/assets/images/logo_baru.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# CSS — Font Awesome + BPJS Theme (ZERO emoji)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');


:root {
    --green:      #00a651;
    --green-dk:   #007a3d;
    --green-lt:   #e8f7ee;
    --green-md:   #c3ebd4;
    --blue:       #003d7c;
    --blue-lt:    #e8f0fb;
    --gold:       #f5a623;
    --gold-lt:    #fff3dc;
    --red:        #e63946;
    --red-lt:     #fdeaea;
    --bg:         #f2f6f3;
    --card:       #ffffff;
    --text:       #1a2d1e;
    --muted:      #5c7363;
    --border:     #d0e8d8;
    --shadow-sm:  0 1px 8px rgba(0,100,50,0.08);
    --shadow-md:  0 4px 20px rgba(0,100,50,0.12);
    --shadow-lg:  0 8px 36px rgba(0,100,50,0.18);
    --r:          12px;
    --r-lg:       18px;
}

*, *::before, *::after {
    font-family: 'Plus Jakarta Sans', sans-serif;
    box-sizing: border-box;
}

html, body, .stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2.5rem 3rem !important;
    max-width: 1300px !important;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(140deg, #002855 0%, #005c2e 55%, #00a651 100%);
    border-radius: 0 0 24px 24px;
    padding: 2.8rem 3.5rem 2.4rem;
    margin: 0 -2.5rem 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    width: 420px; height: 420px;
    top: -120px; right: -80px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 65%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    width: 260px; height: 260px;
    bottom: -60px; left: 30%;
    background: radial-gradient(circle, rgba(0,166,81,0.12) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    color: rgba(255,255,255,0.92);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 99px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-pill i { font-size: 0.78rem; }
.hero-title {
    color: #fff;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.18;
    margin: 0 0 0.5rem;
    letter-spacing: -0.4px;
}
.hero-sub {
    color: rgba(255,255,255,0.75);
    font-size: 0.9rem;
    font-weight: 400;
    margin: 0 0 1.6rem;
    max-width: 580px;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.hstat { text-align: center; }
.hstat-val {
    color: #fff;
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'Space Grotesk', monospace;
}
.hstat-lbl {
    color: rgba(255,255,255,0.65);
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-top: 3px;
}
.hstat-sep {
    width: 1px;
    background: rgba(255,255,255,0.18);
    align-self: stretch;
}

/* ── SECTION HEADING ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 1.2rem;
}
.sec-head-bar {
    width: 4px; height: 22px;
    background: var(--green);
    border-radius: 2px;
    flex-shrink: 0;
}
.sec-head-txt {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--blue);
    margin: 0;
}
.sec-head-count {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--muted);
    background: var(--green-lt);
    border: 1px solid var(--green-md);
    border-radius: 99px;
    padding: 2px 10px;
    margin-left: 4px;
}

/* ── CARD ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-sm);
    padding: 1.4rem 1.6rem;
    transition: box-shadow .2s, transform .2s;
    height: 100%;
}
.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.card-icon-wrap {
    width: 42px; height: 42px;
    background: var(--green-lt);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 12px;
}
.card-icon-wrap i { color: var(--green-dk); font-size: 1.1rem; }
.card-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: var(--muted);
    margin-bottom: 5px;
}
.card-val {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--blue);
    font-family: 'Space Grotesk', monospace;
    line-height: 1;
}
.card-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 5px;
    display: flex;
    align-items: center;
    gap: 4px;
}
.card-sub i { font-size: 0.7rem; }

/* ── TECH CARD ── */
.tech-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-sm);
    padding: 1.3rem 1.1rem 1.1rem;
    text-align: center;
    transition: all .2s;
    cursor: default;
    position: relative;
    overflow: hidden;
}
.tech-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, var(--green));
    border-radius: var(--r-lg) var(--r-lg) 0 0;
}
.tech-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-4px);
    border-color: var(--green);
}
.tech-badge-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 40px;
    margin-bottom: 10px;
}
.tech-badge-wrap img {
    height: 28px;
    object-fit: contain;
}
.tech-name {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--blue);
    margin-bottom: 4px;
}
.tech-desc {
    font-size: 0.67rem;
    color: var(--muted);
    line-height: 1.4;
}

/* ── PRED BOX ── */
.pred-wrap {
    border-radius: var(--r-lg);
    padding: 1.8rem 2rem;
    text-align: center;
    border: 2px solid;
}
.pred-wrap.positif {
    background: linear-gradient(145deg, #e8f7ee, #ccf0da);
    border-color: var(--green);
}
.pred-wrap.netral {
    background: linear-gradient(145deg, #fff3dc, #ffe8b0);
    border-color: var(--gold);
}
.pred-wrap.negatif {
    background: linear-gradient(145deg, #fdeaea, #fccfcf);
    border-color: var(--red);
}
.pred-icon-circle {
    width: 72px; height: 72px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 12px;
    font-size: 1.8rem;
}
.positif .pred-icon-circle { background: #c3ebd4; color: var(--green-dk); }
.netral  .pred-icon-circle { background: #fde8a0; color: #8a6200; }
.negatif .pred-icon-circle { background: #f8b8b8; color: #9b1c25; }
.pred-lbl {
    font-size: 1.7rem;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 4px;
}
.positif .pred-lbl { color: var(--green-dk); }
.netral  .pred-lbl { color: #8a6200; }
.negatif .pred-lbl { color: var(--red); }
.pred-conf {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--muted);
}

/* ── CONFIDENCE BAR ── */
.cbar { margin: 7px 0; }
.cbar-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
}
.cbar-row i { font-size: 0.72rem; margin-right: 4px; }
.cbar-bg {
    background: #e4ede6;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.cbar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .5s ease;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: var(--r);
    padding: 5px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    color: var(--muted) !important;
    padding: 8px 18px !important;
    transition: all .15s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--green) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(0,166,81,0.25) !important;
}

/* ── INPUT ── */
.stTextArea textarea {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-size: 0.88rem !important;
    background: var(--card) !important;
    transition: border-color .2s, box-shadow .2s !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px rgba(0,166,81,0.1) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-dk), var(--green)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.8rem !important;
    width: 100% !important;
    transition: all .2s !important;
    box-shadow: 0 3px 12px rgba(0,166,81,0.22) !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    box-shadow: 0 5px 18px rgba(0,166,81,0.32) !important;
    transform: translateY(-1px) !important;
}

/* ── PILLS ── */
.pill-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.pill {
    background: var(--green-lt);
    border: 1px solid var(--green-md);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--green-dk);
    font-family: 'Space Grotesk', monospace;
}
.pill span { color: var(--muted); font-weight: 400; }

/* ── DIVIDER ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.8rem 0;
}

/* ── PIPELINE STEP ── */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 11px 0;
    border-bottom: 1px solid var(--border);
}
.step-num {
    width: 30px; height: 30px;
    background: var(--green);
    color: white;
    font-size: 0.68rem;
    font-weight: 800;
    font-family: 'Space Grotesk', monospace;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-body-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--blue);
    margin-bottom: 2px;
}
.step-body-desc {
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.5;
}
.step-icon {
    width: 28px; height: 28px;
    background: var(--green-lt);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-icon i { color: var(--green-dk); font-size: 0.78rem; }

/* ── INSIGHT CARD ── */
.insight-card {
    border-radius: var(--r-lg);
    padding: 1.4rem 1.5rem;
    border: 1px solid;
    height: 100%;
}
.insight-icon-wrap {
    width: 44px; height: 44px;
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 12px;
    font-size: 1.2rem;
}

/* ── TABLE ── */
.stDataFrame { border-radius: var(--r) !important; }

/* ── WORDCLOUD WRAPPER ── */
.wc-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-sm);
}
.wc-header {
    padding: 0.9rem 1.4rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}
.wc-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.wc-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--blue);
    margin: 0;
}
.wc-count {
    font-size: 0.72rem;
    color: var(--muted);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 2px 9px;
    margin-left: auto;
}
.wc-body { padding: 1rem 1.4rem 1.2rem; }

/* ── MODEL COMPARE TABLE ── */
.model-compare {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.model-compare th {
    background: var(--green-lt);
    color: var(--blue);
    font-weight: 700;
    padding: 9px 14px;
    text-align: left;
    border-bottom: 2px solid var(--green-md);
}
.model-compare td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.model-compare tr:last-child td { border-bottom: none; }
.model-compare tr:nth-child(even) td { background: var(--bg); }
.winner-badge {
    display: inline-block;
    background: var(--green);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 99px;
    margin-left: 6px;
    vertical-align: middle;
    letter-spacing: 0.4px;
}

/* ── EMPTY STATE ── */
.empty-state {
    border: 1.5px dashed var(--border);
    border-radius: var(--r-lg);
    padding: 2.8rem 2rem;
    text-align: center;
    color: var(--muted);
}
.empty-state i {
    font-size: 2.2rem;
    color: var(--green-md);
    margin-bottom: 12px;
    display: block;
}
.empty-state p {
    font-size: 0.88rem;
    font-weight: 500;
    margin: 0;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)


# INLINE PREPROCESSING (tidak perlu utils/preprocessor.py)


@st.cache_resource
def init_nlp():
    """Inisialisasi semua komponen NLP sekali saja."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stemmer_obj = StemmerFactory().createStemmer()
        sastrawi_sw = set(StopWordRemoverFactory().get_stop_words())
    except Exception:
        stemmer_obj = None
        sastrawi_sw = set()

    custom_sw = {
        'yg','ga','gak','nggak','ngga','udah','udh','sdh','tdk',
        'tp','jg','klo','klu','kalo','dgn','krn','utk',
        'min','kak','mas','mba','mbak','pak','bu',
        'iya','ya','lah','deh','dong','sih','nih','kan',
        'mau','ada','tidak','si','hide','replies','reply','all',
        'bpjs','kesehatan','jkn','halo','hai','hallo',
        'mohon','tolong','bantu','tanya','info',
        'kami','kita','mereka','dia',
    }

    norm_dict = {
        'gak':'tidak','ga':'tidak','nggak':'tidak','ngga':'tidak',
        'udah':'sudah','udh':'sudah','sdh':'sudah',
        'tp':'tapi','krn':'karena','karna':'karena',
        'yg':'yang','dgn':'dengan','utk':'untuk',
        'hrs':'harus','bs':'bisa','blm':'belum',
        'sy':'saya','aq':'saya','ak':'saya',
        'km':'kamu','lu':'kamu','lo':'kamu',
        'klo':'kalau','klu':'kalau','kalo':'kalau',
        'dr':'dari','sm':'sama','aja':'saja','doang':'saja',
        'bgt':'banget','bngt':'banget','jg':'juga',
        'lg':'sedang','lagi':'sedang','skrg':'sekarang',
        'emg':'memang','emang':'memang','gimana':'bagaimana',
        'gmn':'bagaimana','knp':'kenapa',
        'rs':'rumah sakit','faskes':'fasilitas kesehatan',
    }

    return stemmer_obj, sastrawi_sw.union(custom_sw), norm_dict

_stemmer, _stopwords, _norm = init_nlp()

def full_preprocessing(teks: str) -> str:
    """Pipeline preprocessing identik dengan notebook training."""
    from nltk.tokenize import word_tokenize

    teks = str(teks).lower()
    teks = re.sub(r'https?://\S+|www\.\S+', '', teks)
    teks = re.sub(r'@\w+', '', teks)
    teks = re.sub(r'#\w+', '', teks)
    emoji_pat = re.compile(
        u'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
        u'\U00002500-\U00002BEF\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff\u2640-\u2642'
        u'\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+',
        flags=re.UNICODE
    )
    teks = emoji_pat.sub('', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    teks = re.sub(r'\s+', ' ', teks).strip()
    tokens = [_norm.get(t, t) for t in teks.split()]
    teks = ' '.join(tokens)
    try:
        tokens = word_tokenize(teks)
    except Exception:
        tokens = teks.split()
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 1]
    if _stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)


# LOAD MODEL & DATA
@st.cache_resource
def load_models():
    svm_model = joblib.load('model/svm_model.pkl')
    nb_model  = joblib.load('model/nb_model.pkl')
    vectorizer= joblib.load('model/tfidf_vectorizer.pkl')
    encoder   = joblib.load('model/label_encoder.pkl')
    return svm_model, nb_model, vectorizer, encoder

@st.cache_data
def load_data():
    df    = pd.read_csv('data/bpjs_labeled_dataset.csv')
    wc    = json.load(open('data/wordcloud_data.json', encoding='utf-8'))
    stats = json.load(open('data/model_stats.json', encoding='utf-8'))
    return df, wc, stats

try:
    svm_model, nb_model, vectorizer, encoder = load_models()
    df, wc_data, stats = load_data()
    MODEL_OK = True
except Exception as e:
    MODEL_OK   = False
    _err_msg   = str(e)
    stats = {}
    df, wc_data = pd.DataFrame(), {}


# SESSION STATE
if 'history' not in st.session_state:
    st.session_state.history = []
if 'inp' not in st.session_state:
    st.session_state.inp = ''


# HELPERS
def predict(text, use_svm=True):
    processed = full_preprocessing(text)
    if not processed.strip():
        return None, None, None, None
    vec  = vectorizer.transform([processed])
    mdl  = svm_model if use_svm else nb_model
    enc  = mdl.predict(vec)[0]
    proba= mdl.predict_proba(vec)[0]
    lbl  = encoder.inverse_transform([enc])[0]
    pd_  = dict(zip(encoder.classes_, proba))
    return lbl, max(proba), pd_, processed

def sent_icon(label):
    return {
        'Positif': '😊',
        'Netral':  '😐',
        'Negatif': '😠',
    }.get(label, '')

def sent_css(label):
    return label.lower()

def cbar(label, val, color, icon_cls):
    st.markdown(f"""
    <div class="cbar">
        <div class="cbar-row">
            <span><i class="{icon_cls}" style="color:{color}"></i> {label}</span>
            <span style="font-family:'Space Grotesk',monospace;color:{color}">{val*100:.1f}%</span>
        </div>
        <div class="cbar-bg">
            <div class="cbar-fill" style="width:{val*100:.1f}%;background:{color}"></div>
        </div>
    </div>""", unsafe_allow_html=True)

def make_wc(text, cmap):
    if not text or not text.strip():
        return None
    return WordCloud(
        width=900, height=320,
        background_color='white',
        colormap=cmap,
        max_words=90,
        collocations=False,
        random_state=42,
        min_font_size=9,
        max_font_size=80,
        prefer_horizontal=0.8
    ).generate(text)

def sec(title, count=None):
    cnt_html = f'<span class="sec-head-count">{count}</span>' if count else ''
    st.markdown(f"""
    <div class="sec-head">
        <div class="sec-head-bar"></div>
        <p class="sec-head-txt">{title}</p>
        {cnt_html}
    </div>""", unsafe_allow_html=True)


# STAT VARS
total_data = stats.get('total_data', 2432) if MODEL_OK else 2432
label_dist = stats.get('label_dist', {})   if MODEL_OK else {}
acc_svm    = stats.get('svm_accuracy',   0.78) if MODEL_OK else 0.78
f1_svm     = stats.get('svm_f1_macro',   0.73) if MODEL_OK else 0.73
acc_nb     = stats.get('nb_accuracy',    0.67) if MODEL_OK else 0.67
f1_nb      = stats.get('nb_f1_macro',    0.51) if MODEL_OK else 0.51
neg_pct    = stats.get('label_pct', {}).get('Negatif', 41) if MODEL_OK else 41
net_pct    = stats.get('label_pct', {}).get('Netral',  54) if MODEL_OK else 54
pos_pct    = stats.get('label_pct', {}).get('Positif',  4) if MODEL_OK else  4

# HERO
st.markdown(f"""
<div class="hero">
    <div class="hero-pill">
        <i class="fa-solid fa-hospital-user"></i>
        BPJS Kesehatan &nbsp;·&nbsp; NLP Sentiment Analysis
    </div>
    <p class="hero-sub">
        Klasifikasi sentimen berbasis SVM &amp; Naive Bayes.
       Pada Dataset: komentar publik Instagram BPJS Kesehatan.
    </p>
    <div class="hero-stats">
        <div class="hstat">
            <div class="hstat-val">{total_data:,}</div>
            <div class="hstat-lbl">Total Komentar</div>
        </div>
        <div class="hstat-sep"></div>
        <div class="hstat">
            <div class="hstat-val">{acc_svm*100:.1f}%</div>
            <div class="hstat-lbl">Akurasi SVM</div>
        </div>
        <div class="hstat-sep"></div>
        <div class="hstat">
            <div class="hstat-val">{acc_nb*100:.1f}%</div>
            <div class="hstat-lbl">Akurasi Naive Bayes</div>
        </div>
        <div class="hstat-sep"></div>
        <div class="hstat">
            <div class="hstat-val">{f1_svm:.3f}</div>
            <div class="hstat-lbl">F1-Macro SVM</div>
        </div>
        <div class="hstat-sep"></div>
        <div class="hstat">
            <div class="hstat-val">3</div>
            <div class="hstat-lbl">Kelas Sentimen</div>
        </div>
        <div class="hstat-sep"></div>
        <div class="hstat">
            <div class="hstat-val">2</div>
            <div class="hstat-lbl">Model Tersedia</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error(f"**Model tidak dapat dimuat.** Pastikan cell serialisasi notebook sudah dijalankan.\n\n`{_err_msg}`")

# TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "  Prediksi Sentimen",
    "  Analitik Dataset",
    "  Word Cloud",
    "  Tech Stack & Model",
])


# TAB 1 — PREDIKSI

with tab1:
    L, R = st.columns([1.1, 0.9], gap="large")

    with L:
        sec("Input Komentar")

        st.markdown("""
        <p style="font-size:0.76rem;color:var(--muted);margin-bottom:10px;font-weight:500;">
        <i class="fa-solid fa-lightbulb" style="color:var(--gold)"></i>
        &nbsp;Klik contoh komentar untuk isi otomatis:
        </p>""", unsafe_allow_html=True)

        examples = [
            ("Alhamdulillah sudah aktif kembali, terima kasih BPJS!", "pos"),
            ("Pelayanan sangat membantu dan cepat sekali.",             "pos"),
            ("Mau tanya cara daftar BPJS mandiri bagaimana?",          "net"),
            ("Apakah bisa pindah faskes secara online?",               "net"),
            ("Aplikasi error terus tidak bisa login dari kemarin.",     "neg"),
            ("Percuma bayar iuran tapi ditolak saat berobat!",         "neg"),
        ]

        color_map_ex = {'pos': '#e8f7ee', 'net': '#fff3dc', 'neg': '#fdeaea'}
        border_map   = {'pos': '#a8dfc0', 'net': '#f5d078', 'neg': '#f5a8a8'}
        txt_map      = {'pos': '#007a3d', 'net': '#8a6200', 'neg': '#c0392b'}

        cols3 = st.columns(3)
        for idx, (ex, cat) in enumerate(examples):
            with cols3[idx % 3]:
                if st.button(
                    ex[:36] + "…",
                    key=f"ex{idx}",
                    use_container_width=True
                ):
                    st.session_state.inp = ex

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        user_input = st.text_area(
            "komentar",
            value=st.session_state.inp,
            height=120,
            placeholder="Ketik atau tempel komentar Instagram BPJS di sini...",
            label_visibility="collapsed"
        )

        model_choice = st.radio(
            "Pilih model prediksi:",
            [" Support Vector Machine (Terbaik)", "Naive Bayes"],
            horizontal=True,
            label_visibility="visible"
        )
        use_svm = "SVM" in model_choice

        bc1, bc2 = st.columns([2, 1])
        with bc1:
            go = st.button(
                "Analisis Sentimen",
                type="primary",
                use_container_width=True
            )
        with bc2:
            if st.button("Hapus", use_container_width=True):
                st.session_state.inp = ''
                st.rerun()

        if user_input:
            wc = len(user_input.split())
            cc = len(user_input)
            st.markdown(f"""
            <div class="pill-strip">
                <div class="pill"><i class="fa-solid fa-font" style="font-size:.65rem"></i>
                &nbsp;<span>Kata: </span>{wc}</div>
                <div class="pill"><i class="fa-solid fa-text-width" style="font-size:.65rem"></i>
                &nbsp;<span>Karakter: </span>{cc}</div>
            </div>""", unsafe_allow_html=True)

    with R:
        sec("Hasil Prediksi")

        if go and user_input.strip():
            if not MODEL_OK:
                st.error("Model belum dimuat. Jalankan cell serialisasi notebook dulu.")
            else:
                with st.spinner("Menganalisis..."):
                    lbl, conf, pd_, proc = predict(user_input, use_svm)

                if lbl is None:
                    st.warning("Teks terlalu pendek atau kosong setelah preprocessing.")
                else:
                    css = sent_css(lbl)
                    icon_fa = {
                        'Positif': '<i class="a">😊</i>',
                        'Netral':  '<i class="a">😐</i>',
                        'Negatif': '<i class="a">😠</i>',
                    }.get(lbl, '<i class="a">😐</i>')

                    st.markdown(f"""
                    <div class="pred-wrap {css}">
                        <div class="pred-icon-circle">
                            <i class="{icon_fa}"></i>
                        </div>
                        <div class="pred-lbl">{lbl}</div>
                        <div class="pred-conf">
                            <i class="fa-solid fa-chart-simple"></i>
                            &nbsp;Confidence &nbsp;<strong>{conf*100:.1f}%</strong>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

                    st.markdown("""
                    <p style="font-size:0.78rem;font-weight:700;color:var(--blue);margin-bottom:6px;">
                    <i class="fa-solid fa-sliders"></i> &nbsp;Distribusi Probabilitas
                    </p>""", unsafe_allow_html=True)

                    bar_cfg = [
                        ('Positif', '#00a651', 'fa-solid fa-circle-check'),
                        ('Netral',  '#f5a623', 'fa-solid fa-circle-minus'),
                        ('Negatif', '#e63946', 'fa-solid fa-circle-xmark'),
                    ]
                    for bl, bc_, bi in bar_cfg:
                        cbar(bl, pd_.get(bl, 0), bc_, bi)

                    with st.expander("Lihat teks setelah preprocessing"):
                        st.code(proc or "(kosong)", language=None)

                    st.session_state.history.insert(0, {
                        'teks' : user_input[:80] + ('…' if len(user_input) > 80 else ''),
                        'lbl'  : lbl,
                        'conf' : f"{conf*100:.1f}%",
                        'model': "SVM" if use_svm else "NB",
                    })
                    if len(st.session_state.history) > 25:
                        st.session_state.history.pop()
        else:
            st.markdown("""
            <div class="empty-state">
                <i class="fa-regular fa-comment-dots"></i>
                <p>Masukkan komentar dan klik<br>
                <strong style="color:var(--blue)">Analisis Sentimen</strong></p>
            </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        sec("Riwayat Prediksi", f"{len(st.session_state.history)} entri")

        rows = []
        icon_badge = {'Positif': 'POSITIF', 'Netral': 'NETRAL', 'Negatif': 'NEGATIF'}
        for h in st.session_state.history[:15]:
            rows.append({
                'Sentimen'  : h['lbl'],
                'Confidence': h['conf'],
                'Model'     : h['model'],
                'Komentar'  : h['teks'],
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=280
        )
        if st.button("Hapus Riwayat", use_container_width=False):
            st.session_state.history = []
            st.rerun()


# TAB 2 — ANALITIK
with tab2:
    sec("Ringkasan Dataset")

    mc1, mc2, mc3, mc4 = st.columns(4)
    card_data = [
        ("Total Komentar", f"{total_data:,}",
         "Setelah data cleaning",
         "fa-solid fa-comments", "#003d7c"),
        ("Komentar Negatif", f"{label_dist.get('Negatif',0):,}",
         f"{neg_pct}% dari total dataset",
         "fa-solid fa-triangle-exclamation", "#c0392b"),
        ("Komentar Netral", f"{label_dist.get('Netral',0):,}",
         f"{net_pct}% dari total dataset",
         "fa-solid fa-minus-circle", "#8a6200"),
        ("Komentar Positif", f"{label_dist.get('Positif',0):,}",
         f"{pos_pct}% dari total dataset",
         "fa-solid fa-circle-check", "#007a3d"),
    ]
    bg_colors = ['#e8f0fb', '#fdeaea', '#fff3dc', '#e8f7ee']
    ic_colors = ['#003d7c', '#c0392b', '#8a6200', '#007a3d']

    for col, (lbl, val, sub, ico, icc), bg in zip([mc1, mc2, mc3, mc4], card_data, bg_colors):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-icon-wrap" style="background:{bg}">
                    <i class="{ico}" style="color:{icc}"></i>
                </div>
                <div class="card-lbl">{lbl}</div>
                <div class="card-val">{val}</div>
                <div class="card-sub">
                    <i class="fa-solid fa-circle-info"></i> {sub}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    cp, cb = st.columns(2, gap="large")

    with cp:
        sec("Distribusi Sentimen")
        if MODEL_OK and label_dist:
            ordered = ['Negatif', 'Netral', 'Positif']
            sizes   = [label_dist.get(l, 0) for l in ordered]
            clrs    = ['#e63946', '#f5a623', '#00a651']

            fig, ax = plt.subplots(figsize=(5, 4.2))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            wedges, texts, auts = ax.pie(
                sizes, labels=ordered, autopct='%1.1f%%',
                colors=clrs, startangle=90,
                explode=[0.035]*3,
                wedgeprops={'edgecolor':'white','linewidth':2},
                textprops={'fontsize':10,'fontweight':'600','color':'#1a2d1e'}
            )
            for at in auts:
                at.set_fontsize(9); at.set_color('white'); at.set_fontweight('bold')
            ax.set_title('Proporsi Kelas', fontsize=11, fontweight='bold',
                         color='#003d7c', pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with cb:
        sec("Jumlah per Kelas")
        if MODEL_OK and label_dist:
            ordered  = ['Negatif', 'Netral', 'Positif']
            bar_vals = [label_dist.get(l, 0) for l in ordered]
            clrs     = ['#e63946', '#f5a623', '#00a651']

            fig2, ax2 = plt.subplots(figsize=(5, 4.2))
            fig2.patch.set_facecolor('none')
            ax2.set_facecolor('none')
            bars = ax2.bar(ordered, bar_vals, color=clrs, width=0.45,
                           edgecolor='white', linewidth=1.5)
            for b, v in zip(bars, bar_vals):
                ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 8,
                         str(v), ha='center', fontsize=10,
                         fontweight='bold', color='#1a2d1e')
            ax2.set_ylabel('Jumlah Komentar', fontsize=9, color='#5c7363')
            ax2.set_title('Jumlah per Sentimen', fontsize=11, fontweight='bold',
                          color='#003d7c')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#d0e8d8')
            ax2.spines['bottom'].set_color('#d0e8d8')
            ax2.tick_params(colors='#5c7363')
            ax2.grid(axis='y', alpha=0.2, color='#d0e8d8')
            ax2.set_ylim(0, max(bar_vals) * 1.18)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    sec("Top 15 Kata per Sentimen")

    kw3 = st.columns(3)
    kw_cfg = [
        ('Positif', '#00a651', '#007a3d', 'Greens_r'),
        ('Netral',  '#f5a623', '#8a6200', 'YlOrBr_r'),
        ('Negatif', '#e63946', '#9b1c25', 'Reds_r'),
    ]
    if MODEL_OK:
        for col, (sent, bg_c, txt_c, cmap) in zip(kw3, kw_cfg):
            with col:
                txt = wc_data.get(sent, '')
                if txt:
                    top15 = Counter(txt.split()).most_common(15)
                    if top15:
                        wds, cnts = zip(*top15)
                        from matplotlib import colormaps as cmaps
                        fig_k, ax_k = plt.subplots(figsize=(4, 5.5))
                        fig_k.patch.set_facecolor('none')
                        ax_k.set_facecolor('none')
                        c_list = cmaps[cmap](np.linspace(0.25, 0.85, len(top15)))
                        ax_k.barh(list(wds)[::-1], list(cnts)[::-1],
                                  color=c_list, edgecolor='white', height=0.7)
                        for i, (w, cnt) in enumerate(zip(list(wds)[::-1], list(cnts)[::-1])):
                            ax_k.text(cnt + 0.3, i, str(cnt),
                                      va='center', fontsize=7.5, color='#5c7363')
                        ax_k.set_title(sent, fontsize=10, fontweight='bold', color=txt_c)
                        ax_k.set_xlabel('Frekuensi', fontsize=8, color='#5c7363')
                        ax_k.spines['top'].set_visible(False)
                        ax_k.spines['right'].set_visible(False)
                        ax_k.spines['left'].set_color('#d0e8d8')
                        ax_k.spines['bottom'].set_color('#d0e8d8')
                        ax_k.tick_params(labelsize=8, colors='#5c7363')
                        ax_k.grid(axis='x', alpha=0.2, color='#d0e8d8')
                        plt.tight_layout()
                        st.pyplot(fig_k, use_container_width=True)
                        plt.close()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    sec("Sample Dataset Berlabel")

    if MODEL_OK and not df.empty:
        f_sent = st.selectbox(
            "Filter:", ['Semua', 'Positif', 'Netral', 'Negatif'],
            label_visibility='collapsed'
        )
        df_show = df if f_sent == 'Semua' else df[df['sentimen'] == f_sent]
        st.dataframe(
            df_show[['Komentar', 'sentimen']].sample(
                min(20, len(df_show)), random_state=42
            ),
            use_container_width=True,
            hide_index=True,
            height=300,
            column_config={
                'Komentar':  st.column_config.TextColumn('Komentar', width='large'),
                'sentimen':  st.column_config.TextColumn('Sentimen', width='small'),
            }
        )

# TAB 3 — WORD CLOUD
with tab3:
    sec("Word Cloud per Kelas Sentimen")
    st.markdown("""
    <p style="font-size:0.82rem;color:var(--muted);margin:-0.5rem 0 1.2rem">
    <i class="fa-solid fa-circle-info" style="color:var(--green)"></i>
    &nbsp;Ukuran kata merepresentasikan frekuensi kemunculan dalam dataset setelah preprocessing.
    </p>""", unsafe_allow_html=True)

    wc_cfg2 = [
        ('Positif', '#00a651', '#e8f7ee', '#007a3d', 'Greens'),
        ('Netral',  '#f5a623', '#fff3dc', '#8a6200', 'YlOrBr'),
        ('Negatif', '#e63946', '#fdeaea', '#9b1c25', 'Reds'),
    ]

    for sent, dot_c, bg_c, lbl_c, cmap in wc_cfg2:
        n_count = label_dist.get(sent, 0) if MODEL_OK else 0
        st.markdown(f"""
        <div class="wc-section">
            <div class="wc-header" style="background:{bg_c}">
                <div class="wc-dot" style="background:{dot_c}"></div>
                <p class="wc-label">Sentimen {sent}</p>
                <div class="wc-count">{n_count:,} komentar</div>
            </div>
            <div class="wc-body">
        """, unsafe_allow_html=True)

        txt_wc = wc_data.get(sent, '') if MODEL_OK else ''
        if txt_wc and txt_wc.strip():
            wc_obj = make_wc(txt_wc, cmap)
            if wc_obj:
                fig_wc, ax_wc = plt.subplots(figsize=(10, 3.2))
                fig_wc.patch.set_facecolor('none')
                ax_wc.set_facecolor('none')
                ax_wc.imshow(wc_obj, interpolation='bilinear')
                ax_wc.axis('off')
                plt.tight_layout(pad=0.2)
                st.pyplot(fig_wc, use_container_width=True)
                plt.close()
        else:
            st.info("Data tidak tersedia untuk kelas ini.")

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    sec("Word Cloud Kustom")

    cust_txt = st.text_area(
        "Teks kustom:",
        height=90,
        placeholder="Paste teks bebas di sini untuk generate word cloud kustom...",
        label_visibility='collapsed'
    )
    wcc1, wcc2 = st.columns([3, 1])
    with wcc2:
        wc_cmap_sel = st.selectbox(
            "Palet warna:", ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges'],
            label_visibility='collapsed'
        )
    with wcc1:
        gen_wc = st.button("Generate Word Cloud", use_container_width=True)

    if gen_wc and cust_txt.strip():
        proc_wc = full_preprocessing(cust_txt)
        wc_custom = make_wc(proc_wc or cust_txt, wc_cmap_sel)
        if wc_custom:
            st.markdown('<div class="wc-section"><div class="wc-body">', unsafe_allow_html=True)
            fig_c, ax_c = plt.subplots(figsize=(10, 3.2))
            fig_c.patch.set_facecolor('none')
            ax_c.imshow(wc_custom, interpolation='bilinear')
            ax_c.axis('off')
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_c, use_container_width=True)
            plt.close()
            st.markdown("</div></div>", unsafe_allow_html=True)


# TAB 4 — TECH STACK & MODEL
with tab4:
    sec("Tech Stack")
    st.markdown("""
    <p style="font-size:0.8rem;color:var(--muted);margin:-0.5rem 0 1.2rem">
    Library dan framework yang digunakan dalam seluruh pipeline, dari pengumpulan data hingga deployment.
    </p>""", unsafe_allow_html=True)

    tech_data = [
        ("https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat",
         "Python 3.10", "Bahasa utama untuk seluruh pipeline ML", "#3776AB"),
        ("https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white&style=flat",
         "Jupyter Notebook", "Environment eksplorasi dan pengembangan model", "#F37626"),
        ("https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=flat",
         "Pandas", "Manipulasi dan analisis data tabular", "#150458"),
        ("https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=flat",
         "NumPy", "Komputasi numerik dan operasi array", "#013243"),
        ("https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat",
         "Scikit-learn", "Training model, GridSearchCV, evaluasi", "#F7931E"),
        ("https://img.shields.io/badge/TF--IDF-4CAF50?logoColor=white&style=flat",
         "TF-IDF Vectorizer", "Feature extraction: unigram + bigram", "#4CAF50"),
        ("https://img.shields.io/badge/SVM-FF6B35?logoColor=white&style=flat",
         "SVM Linear", "Model klasifikasi utama, class_weight=balanced", "#FF6B35"),
        ("https://img.shields.io/badge/Naive%20Bayes-9C27B0?logoColor=white&style=flat",
         "Naive Bayes", "MultinomialNB, model baseline pembanding", "#9C27B0"),
        ("https://img.shields.io/badge/PySastrawi-2E7D32?logoColor=white&style=flat",
         "PySastrawi", "Stemming & stopword bahasa Indonesia", "#2E7D32"),
        ("https://img.shields.io/badge/NLTK-154f3c?logoColor=white&style=flat",
         "NLTK", "Tokenisasi teks bahasa Indonesia", "#154f3c"),
        ("https://img.shields.io/badge/WordCloud-00BCD4?logoColor=white&style=flat",
         "WordCloud", "Visualisasi frekuensi kata per kelas", "#00BCD4"),
        ("https://img.shields.io/badge/Matplotlib-11557c?logoColor=white&style=flat",
         "Matplotlib", "Visualisasi chart, confusion matrix, learning curve", "#11557c"),
        ("https://img.shields.io/badge/Seaborn-4c8cbf?logoColor=white&style=flat",
         "Seaborn", "Heatmap dan visualisasi statistik lanjutan", "#4c8cbf"),
        ("https://img.shields.io/badge/Joblib-8BC34A?logoColor=white&style=flat",
         "Joblib", "Serialisasi model ke format .pkl", "#8BC34A"),
        ("https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=flat",
         "Streamlit", "Framework deployment dashboard interaktif", "#FF4B4B"),
        ("https://img.shields.io/badge/Regex-607D8B?logoColor=white&style=flat",
         "Regex", "Pattern matching untuk preprocessing & labeling", "#607D8B"),
    ]

    COLS = 4
    for row_start in range(0, len(tech_data), COLS):
        row = tech_data[row_start:row_start + COLS]
        cols = st.columns(COLS)
        for col, (badge_url, name, desc, accent) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="tech-card" style="--accent:{accent}">
                    <div class="tech-badge-wrap">
                        <img src="{badge_url}" alt="{name}" loading="lazy">
                    </div>
                    <div class="tech-name">{name}</div>
                    <div class="tech-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    sec("Perbandingan Model")

    pc1, pc2 = st.columns(2, gap="large")

    with pc1:
        st.markdown("""
        <div class="card">
        """, unsafe_allow_html=True)

        svm_acc = stats.get('svm_accuracy', acc_svm) if MODEL_OK else acc_svm
        svm_f1m = stats.get('svm_f1_macro', f1_svm)  if MODEL_OK else f1_svm
        svm_pre = stats.get('svm_precision', 0.77)    if MODEL_OK else 0.77
        svm_rec = stats.get('svm_recall', 0.78)        if MODEL_OK else 0.78
        nb_acc  = stats.get('nb_accuracy', acc_nb)    if MODEL_OK else acc_nb
        nb_f1m  = stats.get('nb_f1_macro', f1_nb)     if MODEL_OK else f1_nb
        nb_pre  = stats.get('nb_precision', 0.70)     if MODEL_OK else 0.70
        nb_rec  = stats.get('nb_recall', 0.67)         if MODEL_OK else 0.67

        st.markdown(f"""
        <table class="model-compare">
            <thead>
                <tr>
                    <th>Metrik</th>
                    <th>TF-IDF + Support Vector Machine <span class="winner-badge">TERBAIK</span></th>
                    <th>TF-IDF + Naive Bayes</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Accuracy</td>
                    <td><strong>{svm_acc*100:.1f}%</strong></td>
                    <td>{nb_acc*100:.1f}%</td></tr>
                <tr><td>Precision</td>
                    <td><strong>{svm_pre*100:.1f}%</strong></td>
                    <td>{nb_pre*100:.1f}%</td></tr>
                <tr><td>Recall</td>
                    <td><strong>{svm_rec*100:.1f}%</strong></td>
                    <td>{nb_rec*100:.1f}%</td></tr>
                <tr><td>F1-Macro</td>
                    <td><strong>{svm_f1m:.3f}</strong></td>
                    <td>{nb_f1m:.3f}</td></tr>
            </tbody>
        </table>
        </div>""", unsafe_allow_html=True)

    with pc2:
        vocab  = stats.get('vocab_size',   3000) if MODEL_OK else 3000
        tr_sz  = stats.get('train_size',   1945) if MODEL_OK else 1945
        te_sz  = stats.get('test_size',     487) if MODEL_OK else 487
        best_c = stats.get('best_C',        1.0) if MODEL_OK else 1.0
        best_a = stats.get('best_alpha',    0.1) if MODEL_OK else 0.1

        cfg = [
            ("fa-solid fa-sliders",        "Hyperparameter SVM C",  str(best_c)),
            ("fa-solid fa-sliders",        "Hyperparameter NB alpha",str(best_a)),
            ("fa-solid fa-circle-half-stroke","Kernel SVM",          "Linear"),
            ("fa-solid fa-scale-balanced", "Class Weight",           "Balanced"),
            ("fa-solid fa-magnifying-glass","TF-IDF Fitur",          f"{vocab:,}"),
            ("fa-solid fa-grip",           "N-gram Range",           "(1, 2)"),
            ("fa-solid fa-database",       "Training Samples",       f"{tr_sz:,}"),
            ("fa-solid fa-vial",           "Testing Samples",        f"{te_sz:,}"),
            ("fa-solid fa-shuffle",        "Train-Test Split",       "80% / 20%"),
            ("fa-solid fa-layer-group",    "CV Strategy",            "Stratified K-Fold k=5"),
            ("fa-solid fa-bullseye",       "Optimisasi Metrik",      "F1-Macro"),
        ]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-lbl" style="margin-bottom:10px">Konfigurasi Model</p>',
                    unsafe_allow_html=True)
        for ico, k, v in cfg:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 0;border-bottom:1px solid var(--border);font-size:0.8rem;">
                <span style="color:var(--muted);font-weight:500;display:flex;align-items:center;gap:7px;">
                    <i class="{ico}" style="color:var(--green);font-size:0.75rem"></i>{k}
                </span>
                <span style="color:var(--blue);font-weight:700;
                             font-family:'Space Grotesk',monospace;font-size:0.82rem">{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    sec("Alur Pipeline ML")

    steps = [
        ("fa-solid fa-database",        "01", "Data Collection",    "Scraping komentar dari Instagram BPJS Kesehatan menggunakan web scraper"),
        ("fa-solid fa-broom",           "02", "Data Cleaning",      "Hapus duplikat, filter akun resmi BPJS, remove artifact scraping IG"),
        ("fa-solid fa-gears",           "03", "Text Preprocessing", "Case fold → URL/mention/hashtag → emoji → normalisasi → tokenisasi → stemming"),
        ("fa-solid fa-tags",            "04", "Data Labeling",      "7-layer rule-based: whitelist positif, hard negative, sarkasme, negasi, scoring berbobot"),
        ("fa-solid fa-chart-pie",       "05", "EDA",                "Distribusi label, word cloud, bigram/trigram, frekuensi kata per kelas sentimen"),
        ("fa-solid fa-table-list",      "06", "Feature Extraction", "TF-IDF Vectorizer: (1,2)-gram, 3.000 fitur, sublinear TF, min_df=3"),
        ("fa-solid fa-magnifying-glass","07", "GridSearchCV",       "Cari hyperparameter optimal (C dan alpha) via Stratified K-Fold CV=5"),
        ("fa-solid fa-robot",           "08", "Model Training",     "SVM (C=1.0, class_weight=balanced) dan Naive Bayes (alpha=0.1)"),
        ("fa-solid fa-chart-line",      "09", "Evaluation",         "Accuracy, F1-Macro, F1-Weighted, Confusion Matrix, Learning Curve"),
        ("fa-solid fa-floppy-disk",     "10", "Serialization",      "Simpan svm_model.pkl, nb_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl"),
        ("fa-solid fa-rocket",          "11", "Deployment",         "Streamlit dashboard: prediksi real-time, analitik, word cloud, tech stack info"),
    ]

    for ico, num, name, desc in steps:
        st.markdown(f"""
        <div class="step-row">
            <div class="step-icon">
                <i class="{ico}"></i>
            </div>
            <div class="step-num">{num}</div>
            <div style="flex:1">
                <div class="step-body-title">{name}</div>
                <div class="step-body-desc">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    sec("Insight Bisnis")

    ins3 = st.columns(3)
    insights_data = [
        ("fa-solid fa-triangle-exclamation", "#c0392b", "#fdeaea", "#ffd5d5",
         "Sentimen Dominan",
         f"Negatif {neg_pct}%",
         "Mayoritas komentar berupa keluhan — gap signifikan antara ekspektasi peserta dan kualitas layanan BPJS aktual."),
        ("fa-solid fa-clipboard-list", "#8a6200", "#fff3dc", "#fce8a0",
         "Isu Utama Publik",
         "Administrasi & Teknis",
         "Top keluhan: aplikasi error, BPJS nonaktif, ditolak RS, tagihan tidak sesuai. Dua kluster isu yang membutuhkan respons berbeda."),
        ("fa-solid fa-lightbulb", "#007a3d", "#e8f7ee", "#b8eacc",
         "Rekomendasi",
         "Monitoring Berkelanjutan",
         "Integrasikan model ke social media monitoring untuk deteksi dini keluhan viral sebelum berkembang menjadi krisis reputasi."),
    ]

    for col, (ico, col_c, bg_c, ico_bg, title, sub, desc) in zip(ins3, insights_data):
        with col:
            st.markdown(f"""
            <div class="insight-card" style="background:{bg_c};border-color:{col_c}40">
                <div class="insight-icon-wrap" style="background:{ico_bg}">
                    <i class="{ico}" style="color:{col_c}"></i>
                </div>
                <p style="font-size:0.7rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.8px;color:{col_c};margin:0 0 4px">{title}</p>
                <p style="font-size:1rem;font-weight:800;color:{col_c};margin:0 0 8px;
                          font-family:'Space Grotesk',sans-serif">{sub}</p>
                <p style="font-size:0.77rem;color:var(--muted);margin:0;line-height:1.55">{desc}</p>
            </div>""", unsafe_allow_html=True)


# FOOTER
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;color:var(--muted);font-size:0.77rem;">
    <i class="fa-solid fa-hospital-user" style="color:var(--green);margin-right:6px"></i>
    <strong style="color:var(--green)">BPJS Sentiment Analyzer</strong>
    &nbsp;&middot;&nbsp; Capstone Project
    &nbsp;&middot;&nbsp; NLP &amp; Machine Learning
    &nbsp;&middot;&nbsp; Data Science &amp; Deployment
    &nbsp;&middot;&nbsp; CC26-PRU476
</div>
""", unsafe_allow_html=True)
