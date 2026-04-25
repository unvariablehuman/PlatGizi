import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import os
import html as _h
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PlatGizi – Smart Menu Planner MBG",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.main { background-color: #f8fdf4; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: visible;}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a6b3c 0%, #145530 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 8px 0 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

/* ── HERO ── */
.hero-box {
    background: linear-gradient(135deg, #1a6b3c 0%, #2d9e5f 50%, #52c788 100%);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(26,107,60,0.25);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: white;
    margin: 0;
    letter-spacing: -1px;
}
.hero-subtitle { font-size: 1.1rem; color: rgba(255,255,255,0.85); margin-top: 8px; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    color: white;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.85rem;
    margin-top: 12px;
}

/* ── SECTION TITLE ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #1a6b3c;
    border-left: 5px solid #2d9e5f;
    padding-left: 16px;
    margin-bottom: 20px;
}
.section-desc {
    background: white;
    border-radius: 12px;
    padding: 18px 22px;
    border: 1px solid #e8f5ee;
    color: #555;
    line-height: 1.7;
    margin-bottom: 24px;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 2px solid #e8f5ee;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.metric-num { font-size: 2rem; font-weight: 900; color: #1a6b3c; line-height: 1; }
.metric-unit { font-size: 0.75rem; color: #888; margin-top: 2px; }
.metric-label { font-size: 0.85rem; color: #555; font-weight: 700; margin-top: 6px; }

/* ── INFO CARDS ── */
.info-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #e8f5ee;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.info-card h4 { color: #1a6b3c; margin-bottom: 8px; font-size: 1rem; }

/* ── CLUSTER CHIPS ── */
.chip {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 700;
    margin: 3px;
    color: white;
}

/* ── PROFIL CARD ── */
.profil-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    border: 2px solid #e8f5ee;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 24px;
}
.profil-card h3 { color: #1a6b3c; font-size: 1.2rem; margin-bottom: 4px; }

/* ── MEAL CARDS ── */
.meal-card {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e0f0e6;
    overflow: hidden;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(26,107,60,0.07);
}
.meal-card-header { padding: 10px 16px; font-weight: 800; font-size: 0.95rem; color: white; }
.meal-card-body { display: flex; }
.meal-col { flex: 1; padding: 12px 14px; border-right: 1px solid #f0f7f2; }
.meal-col:last-child { border-right: none; }
.meal-col-label { font-size: 0.68rem; font-weight: 800; color: #607566; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 5px; }
.meal-col-name { font-size: 0.88rem; font-weight: 600; color: #1d2b22; line-height: 1.4; }
.meal-gizi-bar { background: #f4fbf6; padding: 7px 16px; font-size: 0.77rem; color: #607566; border-top: 1px solid #e8f5ee; font-weight: 600; }

/* ── GIZI PANEL ── */
.gizi-panel { background: #f8fdf4; border-radius: 14px; padding: 16px 18px; border: 1px solid #e8f5ee; }
.gizi-panel-title { font-size: 0.92rem; font-weight: 800; color: #1a6b3c; margin-bottom: 14px; }

/* ── STAT BOX ── */
.stat-box { background: white; border-radius: 12px; padding: 16px; text-align: center; border: 2px solid #e8f5ee; }
.stat-number { font-size: 1.8rem; font-weight: 900; color: #1a6b3c; line-height: 1; }
.stat-unit { font-size: 0.8rem; color: #888; margin-top: 2px; }
.stat-label { font-size: 0.85rem; color: #555; font-weight: 600; margin-top: 4px; }

/* ── STEP BADGE ── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a6b3c, #2d9e5f);
    color: white;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    line-height: 36px;
    text-align: center;
    font-weight: 900;
    font-size: 1rem;
    margin-right: 10px;
    vertical-align: middle;
}

/* ── EVAL TABLE ── */
.eval-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.eval-table th {
    background: #1a6b3c;
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 700;
}
.eval-table td { padding: 10px 14px; border-bottom: 1px solid #e8f5ee; }
.eval-table tr:nth-child(even) td { background: #f8fdf4; }

div.stButton > button {
    background: linear-gradient(135deg, #1a6b3c, #2d9e5f);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 16px rgba(26,107,60,0.3);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(26,107,60,0.4);
}
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.8rem;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #e8f5ee;
}
.main .block-container { padding-bottom: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_recommender():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'recommender.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_images():
    base = os.path.dirname(os.path.abspath(__file__))
    imgs = {}
    for name in ['eda_distribusi', 'eda_top10', 'eda_korelasi', 'eda_resep',
                 'cluster_plot', 'elbow_plot']:
        p = os.path.join(base, f'{name}.png')
        if os.path.exists(p):
            imgs[name] = p
    return imgs

try:
    rec          = load_recommender()
    nutrition_df = rec['nutrition_df']
    resep_df     = rec['resep_df']
    scaler       = rec['scaler']
    kmeans       = rec['kmeans']
    PROFIL_GIZI  = rec['profil_gizi']
    model_loaded = True
except Exception as e:
    model_loaded = False

imgs = load_images()


# ─────────────────────────────────────────
# FUNGSI REKOMENDASI
# ─────────────────────────────────────────
def cari_makanan_mirip(target_kalori, target_protein, target_lemak, target_karbo,
                       cluster_id, n=20, exclude=[]):
    subset = nutrition_df[
        (nutrition_df['cluster'] == cluster_id) &
        (~nutrition_df['name'].isin(exclude))
    ].copy()
    if len(subset) == 0:
        return pd.DataFrame()
    target_vector        = scaler.transform([[target_kalori, target_protein, target_lemak, target_karbo]])
    food_vectors         = subset[['calories_norm', 'proteins_norm', 'fat_norm', 'carbo_norm']].values
    subset['similarity'] = cosine_similarity(target_vector, food_vectors)[0]
    return subset.nlargest(n, 'similarity')


def generate_menu_harian(profil, seed=None):
    if seed is not None:
        random.seed(seed)
    target     = PROFIL_GIZI[profil]
    proporsi   = {'Sarapan': 0.25, 'Makan Siang': 0.40, 'Makan Malam': 0.35}
    menu       = {}
    used_karbo = []
    used_prot  = []
    used_sayur = []
    total      = {'kalori': 0, 'protein': 0, 'lemak': 0, 'karbo': 0}

    for waktu, pct in proporsi.items():
        tk = target['kalori']  * pct
        tp = target['protein'] * pct
        tl = target['lemak']   * pct
        tc = target['karbo']   * pct

        karbo_cand = cari_makanan_mirip(tk*0.5, tp*0.2, tl*0.2, tc*0.7, 2, exclude=used_karbo)
        prot_cand  = cari_makanan_mirip(tk*0.4, tp*0.6, tl*0.5, tc*0.2, 3, exclude=used_prot)
        sayur_cand = cari_makanan_mirip(tk*0.1, tp*0.2, tl*0.1, tc*0.1, 0, exclude=used_sayur)

        karbo_cand = karbo_cand[karbo_cand['calories'] <= tk * 0.7] if len(karbo_cand) > 0 else karbo_cand
        prot_cand  = prot_cand[prot_cand['calories']   <= tk * 0.6] if len(prot_cand)  > 0 else prot_cand
        sayur_cand = sayur_cand[sayur_cand['calories'] <= tk * 0.3] if len(sayur_cand) > 0 else sayur_cand

        if len(karbo_cand) == 0: karbo_cand = cari_makanan_mirip(tk*0.5, tp*0.2, tl*0.2, tc*0.7, 2, exclude=used_karbo)
        if len(prot_cand)  == 0: prot_cand  = cari_makanan_mirip(tk*0.4, tp*0.6, tl*0.5, tc*0.2, 3, exclude=used_prot)
        if len(sayur_cand) == 0: sayur_cand = cari_makanan_mirip(tk*0.1, tp*0.2, tl*0.1, tc*0.1, 0, exclude=used_sayur)

        karbo = karbo_cand.sample(1).iloc[0] if len(karbo_cand) > 0 else None
        prot  = prot_cand.sample(1).iloc[0]  if len(prot_cand)  > 0 else None
        sayur = sayur_cand.sample(1).iloc[0] if len(sayur_cand) > 0 else None

        if karbo is not None: used_karbo.append(karbo['name'])
        if prot  is not None: used_prot.append(prot['name'])
        if sayur is not None: used_sayur.append(sayur['name'])

        gizi = {
            'kalori' : (karbo['calories']     if karbo is not None else 0) + (prot['calories']      if prot  is not None else 0) + (sayur['calories']     if sayur is not None else 0),
            'protein': (karbo['proteins']     if karbo is not None else 0) + (prot['proteins']      if prot  is not None else 0) + (sayur['proteins']     if sayur is not None else 0),
            'lemak'  : (karbo['fat']          if karbo is not None else 0) + (prot['fat']           if prot  is not None else 0) + (sayur['fat']          if sayur is not None else 0),
            'karbo'  : (karbo['carbohydrate'] if karbo is not None else 0) + (prot['carbohydrate']  if prot  is not None else 0) + (sayur['carbohydrate'] if sayur is not None else 0),
        }
        menu[waktu] = {
            'karbo'  : karbo['name'] if karbo is not None else '-',
            'protein': prot['name']  if prot  is not None else '-',
            'sayur'  : sayur['name'] if sayur is not None else '-',
            'gizi'   : gizi,
        }
        for k in total:
            total[k] += gizi[k]

    return {'profil': profil, 'target': target, 'menu': menu, 'total': {k: round(v, 1) for k, v in total.items()}}


def generate_menu_mingguan(profil, jumlah_hari=7):
    return [{**generate_menu_harian(profil, seed=hari * 13), 'hari': hari + 1} for hari in range(jumlah_hari)]


# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-family:'Playfair Display',serif; font-size:1.6rem; color:white; font-weight:700;">🍽️ PlatGizi</div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.7); margin-top:4px;">Smart Menu Planner MBG</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        options=[
            "🏠  Home",
            "📊  Step 1: EDA",
            "⚙️  Step 2: Preprocessing",
            "🔵  Step 3: K-Means Clustering",
            "🎯  Step 4: Content-Based Filtering",
            "🍽️  Step 5: Demo Rekomendasi",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:rgba(255,255,255,0.55); text-align:center; line-height:1.8;">
        Machine Learning Project<br>
        Binus University · LC01<br>
        SDG 2 🌱 & SDG 3 💚
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────
def hero(title, subtitle, badge="🤖 Powered by Machine Learning"):
    st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        <div class="hero-badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════
if page == "🏠  Home":
    hero("🍽️ PlatGizi", "Smart Menu Planner untuk Program Makan Bergizi Gratis (MBG)")

    col1, col2 = st.columns([3, 2])
    with col1:
        section("Tentang Project")
        st.markdown("""
        <div class="section-desc">
            <b>PlatGizi</b> adalah sistem rekomendasi menu makanan harian berbasis Machine Learning yang dirancang
            untuk mendukung Program <b>Makan Bergizi Gratis (MBG)</b> dari pemerintah Indonesia.<br><br>
            Sistem ini menggabungkan dua algoritma utama:
            <ul>
                <li><b>K-Means Clustering</b> — mengelompokkan makanan berdasarkan profil gizinya</li>
                <li><b>Content-Based Filtering</b> — merekomendasikan kombinasi menu yang sesuai target kalori & gizi per profil pengguna</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        section("Dataset yang Digunakan")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-card">
                <h4>🥗 Kandungan Gizi Makanan Indonesia</h4>
                <div style="font-size:0.85rem;color:#555;">Sumber: Kaggle<br>
                <b style="color:#1a6b3c;">~1.345</b> jenis makanan dengan data kalori, protein, lemak, & karbohidrat</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-card">
                <h4>🍳 Resep Masakan Indonesia</h4>
                <div style="font-size:0.85rem;color:#555;">Sumber: Kaggle<br>
                <b style="color:#1a6b3c;">~16.000</b> resep dari 8 kategori bahan utama</div>
            </div>
            """, unsafe_allow_html=True)

        section("SDGs yang Relevan")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("""
            <div class="info-card" style="border-left:4px solid #f5a623;">
                <h4>🎯 SDG 2 – Zero Hunger</h4>
                <div style="font-size:0.85rem;color:#555;">Memastikan akses terhadap makanan bergizi yang cukup untuk semua kalangan</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
            <div class="info-card" style="border-left:4px solid #3498db;">
                <h4>🎯 SDG 3 – Good Health</h4>
                <div style="font-size:0.85rem;color:#555;">Mendukung kesehatan dan kesejahteraan melalui pola makan bergizi seimbang</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        section("Alur Project ML")
        steps = [
            ("1", "📊 EDA", "Eksplorasi data mentah sebelum preprocessing"),
            ("2", "⚙️ Preprocessing", "Bersihkan & siapkan kedua dataset"),
            ("3", "🔵 K-Means Clustering", "Kelompokkan makanan berdasarkan profil gizi"),
            ("4", "🎯 Content-Based Filtering", "Rekomendasikan menu sesuai target kalori/gizi"),
            ("5", "🍽️ Demo", "Tampilkan hasil sebagai aplikasi web interaktif"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="info-card" style="padding:14px 18px; margin-bottom:10px;">
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <span class="step-badge">{num}</span>
                    <b style="color:#1a6b3c;">{title}</b>
                </div>
                <div style="font-size:0.82rem;color:#666;padding-left:46px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        section("Target Pengguna")
        profiles = [
            ("🧒", "SD Kelas 1–3", "1.400 kkal/hari"),
            ("👦", "SD Kelas 4–6", "1.600 kkal/hari"),
            ("🎓", "SMP/SMA", "2.000 kkal/hari"),
            ("🤱", "Ibu Hamil/Menyusui", "2.200 kkal/hari"),
        ]
        for icon, label, kal in profiles:
            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:8px 14px;background:white;
                        border-radius:10px;border:1px solid #e8f5ee;margin-bottom:8px;">
                <span style="font-size:1.3rem;margin-right:12px;">{icon}</span>
                <div>
                    <div style="font-weight:700;color:#1d2b22;font-size:0.9rem;">{label}</div>
                    <div style="font-size:0.78rem;color:#888;">{kal}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════
elif page == "📊  Step 1: EDA":
    hero("📊 Exploratory Data Analysis", "Memahami karakteristik data sebelum preprocessing", "🔍 Step 1 dari 5")

    st.markdown("""
    <div class="section-desc">
        EDA dilakukan untuk memahami <b>distribusi, pola, dan karakteristik</b> data mentah sebelum masuk ke tahap preprocessing.
        Dengan EDA, kita bisa mengidentifikasi nilai kosong, outlier, dan insight awal yang berguna untuk proses selanjutnya.
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model tidak berhasil dimuat.")
        st.stop()

    # ── Statistik Deskriptif ──
    section("📋 Statistik Deskriptif Dataset Gizi")
    stats = nutrition_df[['calories', 'proteins', 'fat', 'carbohydrate']].describe().round(2)
    stats.index = ['Count', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max']
    stats.columns = ['Kalori (kkal)', 'Protein (g)', 'Lemak (g)', 'Karbohidrat (g)']

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, f"{int(nutrition_df['calories'].mean())}", "kkal", "Rata-rata Kalori"),
        (c2, f"{nutrition_df['proteins'].mean():.1f}", "gram", "Rata-rata Protein"),
        (c3, f"{nutrition_df['fat'].mean():.1f}", "gram", "Rata-rata Lemak"),
        (c4, f"{nutrition_df['carbohydrate'].mean():.1f}", "gram", "Rata-rata Karbo"),
    ]
    for col, num, unit, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-num">{num}</div>
                <div class="metric-unit">{unit}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(stats, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-card">
            <h4>🥗 Dataset Gizi</h4>
            <div style="font-size:0.85rem;color:#555;">
                Total makanan: <b style="color:#1a6b3c;">{len(nutrition_df)}</b> item<br>
                Kalori min: <b>{nutrition_df['calories'].min():.0f}</b> kkal &nbsp;|&nbsp;
                Kalori max: <b>{nutrition_df['calories'].max():.0f}</b> kkal
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if 'kategori' in resep_df.columns:
            st.markdown(f"""
            <div class="info-card">
                <h4>🍳 Dataset Resep</h4>
                <div style="font-size:0.85rem;color:#555;">
                    Total resep: <b style="color:#1a6b3c;">{len(resep_df)}</b> item<br>
                    Kategori: <b>{resep_df['kategori'].nunique()}</b> jenis bahan utama
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Visualisasi ──
    section("📈 Distribusi Nilai Gizi Makanan")
    if 'eda_distribusi' in imgs:
        st.image(imgs['eda_distribusi'], use_container_width=True)
    else:
        st.info("📁 File `eda_distribusi.png` belum ditemukan. Pastikan file ada di folder yang sama dengan `app.py`.")

    section("🏆 Top 10 Makanan Tertinggi per Nutrisi")
    if 'eda_top10' in imgs:
        st.image(imgs['eda_top10'], use_container_width=True)
    else:
        st.info("📁 File `eda_top10.png` belum ditemukan.")

    col1, col2 = st.columns(2)
    with col1:
        section("🔥 Korelasi Antar Nutrisi")
        if 'eda_korelasi' in imgs:
            st.image(imgs['eda_korelasi'], use_container_width=True)
        else:
            st.info("📁 File `eda_korelasi.png` belum ditemukan.")

    with col2:
        section("🍳 Distribusi Resep per Kategori")
        if 'eda_resep' in imgs:
            st.image(imgs['eda_resep'], use_container_width=True)
        else:
            st.info("📁 File `eda_resep.png` belum ditemukan.")

    # ── Insight ──
    section("💡 Insight dari EDA")
    i1, i2, i3 = st.columns(3)
    corr = nutrition_df[['calories','proteins','fat','carbohydrate']].corr()
    with i1:
        st.markdown(f"""
        <div class="info-card" style="border-left:4px solid #e74c3c;">
            <h4>📊 Distribusi Kalori</h4>
            <div style="font-size:0.85rem;color:#555;">
                Data kalori <b>right-skewed</b> — sebagian besar makanan memiliki kalori rendah-sedang,
                dengan beberapa outlier tinggi. Median ({nutrition_df['calories'].median():.0f} kkal) &lt; Mean ({nutrition_df['calories'].mean():.0f} kkal).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="info-card" style="border-left:4px solid #f39c12;">
            <h4>🔗 Korelasi Gizi</h4>
            <div style="font-size:0.85rem;color:#555;">
                Korelasi kalori–lemak: <b>{corr.loc['calories','fat']:.2f}</b><br>
                Korelasi kalori–karbo: <b>{corr.loc['calories','carbohydrate']:.2f}</b><br>
                Lemak berkontribusi lebih besar terhadap kalori dibanding karbo.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="info-card" style="border-left:4px solid #3498db;">
            <h4>🍳 Variasi Resep</h4>
            <div style="font-size:0.85rem;color:#555;">
                Dataset resep mencakup 8 kategori protein hewani & nabati.
                Variasi ini memungkinkan sistem merekomendasikan menu yang beragam dan tidak monoton.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: PREPROCESSING
# ═══════════════════════════════════════════
elif page == "⚙️  Step 2: Preprocessing":
    hero("⚙️ Data Preprocessing", "Membersihkan & menyiapkan data untuk model ML", "🔧 Step 2 dari 5")

    st.markdown("""
    <div class="section-desc">
        Preprocessing adalah tahap <b>membersihkan dan menyiapkan data</b> sebelum dimasukkan ke model Machine Learning.
        Tanpa preprocessing yang baik, model ML bisa menghasilkan rekomendasi yang tidak akurat atau error.
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model tidak berhasil dimuat.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        section("🥗 Preprocessing Dataset Gizi")
        steps_gizi = [
            ("Hapus kolom tidak dipakai", "Kolom `id` dan `image` dihapus karena tidak relevan untuk ML"),
            ("Hapus baris gizi = 0", "Baris yang semua nilai gizinya 0 dianggap data tidak valid"),
            ("Isi nilai kosong (NaN)", "NaN diisi dengan nilai median agar tidak bias ke mean"),
            ("Hapus duplikat nama", "Makanan dengan nama sama dihapus untuk menghindari redundansi"),
            ("Normalisasi (MinMaxScaler)", "Skala nilai gizi ke 0–1 agar kalori tidak mendominasi fitur lain"),
        ]
        for i, (title, desc) in enumerate(steps_gizi, 1):
            st.markdown(f"""
            <div class="info-card" style="padding:12px 16px;margin-bottom:10px;">
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <span style="background:#1a6b3c;color:white;border-radius:50%;
                                 width:24px;height:24px;line-height:24px;text-align:center;
                                 font-size:0.75rem;font-weight:900;margin-right:10px;flex-shrink:0;">{i}</span>
                    <b style="color:#1a6b3c;font-size:0.9rem;">{title}</b>
                </div>
                <div style="font-size:0.82rem;color:#666;padding-left:34px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # Stats before/after
        st.markdown("""
        <div class="info-card" style="background:#f8fdf4;">
            <h4>📊 Hasil Preprocessing Gizi</h4>
        """, unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Data Awal", "~1.345 baris")
        with mc2:
            st.metric("Data Bersih", f"{len(nutrition_df)} baris")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        section("🍳 Preprocessing Dataset Resep")
        steps_resep = [
            ("Gabungkan 8 file CSV", "Semua file per kategori (ayam, ikan, tempe, dll) digabung jadi 1 DataFrame"),
            ("Tambah kolom kategori", "Kolom `kategori` ditambahkan sebagai label asal file"),
            ("Hapus duplikat judul", "Resep dengan judul sama dihapus"),
            ("Hapus baris kosong", "Baris yang Title atau Ingredients kosong dihapus"),
            ("Seleksi kolom penting", "Hanya simpan kolom: Title, Ingredients, Loves, kategori"),
        ]
        for i, (title, desc) in enumerate(steps_resep, 1):
            st.markdown(f"""
            <div class="info-card" style="padding:12px 16px;margin-bottom:10px;">
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <span style="background:#2d9e5f;color:white;border-radius:50%;
                                 width:24px;height:24px;line-height:24px;text-align:center;
                                 font-size:0.75rem;font-weight:900;margin-right:10px;flex-shrink:0;">{i}</span>
                    <b style="color:#1a6b3c;font-size:0.9rem;">{title}</b>
                </div>
                <div style="font-size:0.82rem;color:#666;padding-left:34px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style="background:#f8fdf4;">
            <h4>📊 Hasil Preprocessing Resep</h4>
        """, unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Data Awal", "~16.000+ baris")
        with mc2:
            st.metric("Data Bersih", f"{len(resep_df)} baris")
        st.markdown("</div>", unsafe_allow_html=True)

    # Sample data
    section("🔍 Preview Data Hasil Preprocessing")
    tab1, tab2 = st.tabs(["Dataset Gizi (nutrition_clean)", "Dataset Resep (resep_clean)"])
    with tab1:
        show_cols = ['name','calories','proteins','fat','carbohydrate','calories_norm','proteins_norm','fat_norm','carbo_norm']
        show_cols = [c for c in show_cols if c in nutrition_df.columns]
        st.dataframe(nutrition_df[show_cols].head(10), use_container_width=True)
    with tab2:
        show_cols2 = [c for c in ['Title','Ingredients','Loves','kategori'] if c in resep_df.columns]
        st.dataframe(resep_df[show_cols2].head(10), use_container_width=True)

    section("💡 Mengapa Normalisasi Penting?")
    st.markdown("""
    <div class="section-desc">
        Kalori memiliki rentang nilai <b>0–1000+ kkal</b>, sedangkan protein/lemak/karbo berada di <b>0–100 gram</b>.
        Tanpa normalisasi, kalori akan <b>mendominasi perhitungan jarak</b> di K-Means, sehingga cluster hanya mencerminkan perbedaan kalori,
        bukan profil gizi secara keseluruhan. Dengan <b>MinMaxScaler</b>, semua fitur berada di skala 0–1 sehingga kontribusinya seimbang.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: CLUSTERING
# ═══════════════════════════════════════════
elif page == "🔵  Step 3: K-Means Clustering":
    hero("🔵 K-Means Clustering", "Mengelompokkan makanan berdasarkan profil gizinya", "📐 Step 3 dari 5")

    st.markdown("""
    <div class="section-desc">
        <b>K-Means</b> adalah algoritma unsupervised learning yang mengelompokkan data menjadi <b>K cluster</b>
        berdasarkan kemiripan karakteristiknya. Pada project ini, 1.346 makanan dikelompokkan berdasarkan
        4 fitur gizi: kalori, protein, lemak, dan karbohidrat (setelah dinormalisasi).
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model tidak berhasil dimuat.")
        st.stop()

    col1, col2 = st.columns([3, 2])

    with col1:
        section("📐 Pemilihan K Optimal (Elbow Method)")
        st.markdown("""
        <div class="section-desc" style="margin-bottom:12px;">
            Elbow Method mencari nilai K terbaik dengan melihat titik "siku" — di mana penambahan cluster
            tidak lagi signifikan mengurangi inertia (total jarak dalam cluster). K=4 dipilih karena merupakan titik elbow yang paling jelas.
        </div>
        """, unsafe_allow_html=True)
        if 'elbow_plot' in imgs:
            st.image(imgs['elbow_plot'], use_container_width=True)
        else:
            st.info("📁 File `elbow_plot.png` belum ditemukan.")

        section("🔵 Visualisasi Hasil Clustering")
        if 'cluster_plot' in imgs:
            st.image(imgs['cluster_plot'], use_container_width=True)
        else:
            st.info("📁 File `cluster_plot.png` belum ditemukan.")

    with col2:
        section("📊 Evaluasi K-Means (K=4)")
        evals = [
            ("Silhouette Score", "0.5039", "> 0.5 = Good", "#2d9e5f", "Mengukur seberapa mirip data dalam cluster vs cluster lain"),
            ("Davies-Bouldin Index", "0.8478", "< 1.0 = Baik", "#3498db", "Rasio jarak dalam cluster vs antar cluster (makin kecil makin baik)"),
            ("Calinski-Harabasz", "1031.01", "Makin tinggi makin baik", "#9b59b6", "Perbandingan dispersi antar cluster vs dalam cluster"),
        ]
        for name, val, status, color, desc in evals:
            st.markdown(f"""
            <div class="info-card" style="border-left:4px solid {color}; margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <b style="color:#1d2b22;font-size:0.9rem;">{name}</b>
                    <span style="font-size:1.4rem;font-weight:900;color:{color};">{val}</span>
                </div>
                <div style="font-size:0.78rem;color:white;background:{color};
                            border-radius:8px;padding:2px 10px;display:inline-block;margin-bottom:6px;">{status}</div>
                <div style="font-size:0.8rem;color:#666;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        section("🏷️ Profil 4 Cluster")
        clusters = [
            ("0", "🟢 Rendah Kalori", "#2d9e5f", "Rendah di semua nutrisi — sayuran, lalapan, buah segar"),
            ("1", "🔴 Tinggi Lemak", "#e74c3c", "Kalori & lemak sangat tinggi — gorengan, santan, daging berlemak"),
            ("2", "🟡 Tinggi Karbo", "#f39c12", "Karbo tinggi, kalori sedang — nasi, ubi, singkong, roti"),
            ("3", "🔵 Tinggi Protein", "#3498db", "Protein tinggi, kalori sedang — ikan, ayam, tahu, tempe, telur"),
        ]
        for num, name, color, desc in clusters:
            st.markdown(f"""
            <div class="info-card" style="padding:12px 14px;margin-bottom:8px;">
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <span style="background:{color};color:white;border-radius:50%;width:22px;height:22px;
                                 line-height:22px;text-align:center;font-size:0.7rem;font-weight:900;
                                 margin-right:8px;flex-shrink:0;">{num}</span>
                    <b style="color:#1d2b22;font-size:0.88rem;">{name}</b>
                </div>
                <div style="font-size:0.8rem;color:#666;padding-left:30px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Distribusi cluster
    if 'cluster' in nutrition_df.columns:
        section("📊 Distribusi Makanan per Cluster")
        cluster_names = {0: "🟢 Rendah Kalori", 1: "🔴 Tinggi Lemak", 2: "🟡 Tinggi Karbo", 3: "🔵 Tinggi Protein"}
        dist = nutrition_df['cluster'].value_counts().sort_index()
        cc = st.columns(4)
        for i, (cidx, count) in enumerate(dist.items()):
            with cc[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-num">{count}</div>
                    <div class="metric-unit">makanan</div>
                    <div class="metric-label">{cluster_names.get(cidx, f'Cluster {cidx}')}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: CONTENT-BASED FILTERING
# ═══════════════════════════════════════════
elif page == "🎯  Step 4: Content-Based Filtering":
    hero("🎯 Content-Based Filtering", "Merekomendasikan menu sesuai target kalori & gizi", "🤖 Step 4 dari 5")

    st.markdown("""
    <div class="section-desc">
        <b>Content-Based Filtering</b> merekomendasikan item berdasarkan <b>kesamaan karakteristik konten</b>.
        Pada project ini, sistem mencari makanan yang paling mirip dengan target gizi pengguna menggunakan
        <b>Cosine Similarity</b> sebagai metrik kesamaan.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        section("⚙️ Cara Kerja Sistem")
        st.markdown("""
        <div class="info-card">
            <div style="font-family:monospace;font-size:0.85rem;color:#1d2b22;line-height:2.2;background:#f8fdf4;
                        padding:16px;border-radius:8px;border:1px solid #e8f5ee;">
                User pilih profil (Anak SD / SMP / Ibu Menyusui)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                Sistem tentukan <b>target kalori & gizi harian</b><br>
                &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                Bagi target per waktu makan (25% / 40% / 35%)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                Cosine Similarity cari makanan paling mirip target<br>
                &nbsp;&nbsp;&nbsp;&nbsp;dari cluster yang sesuai (Karbo / Protein / Sayuran)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                Kombinasikan jadi <b>menu harian yang seimbang</b><br>
                &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                Ulangi untuk 7 hari (tanpa duplikasi makanan dalam 1 hari)
            </div>
        </div>
        """, unsafe_allow_html=True)

        section("🔢 Cosine Similarity")
        st.markdown("""
        <div class="section-desc">
            Cosine Similarity mengukur sudut antara dua vektor gizi — semakin kecil sudutnya (nilai mendekati 1),
            semakin mirip profil gizi makanan dengan target pengguna. Formula:<br><br>
            <div style="text-align:center;font-size:1.1rem;font-weight:700;color:#1a6b3c;padding:12px;
                        background:white;border-radius:8px;border:2px solid #e8f5ee;">
                similarity = (A · B) / (‖A‖ × ‖B‖)
            </div><br>
            Di mana <b>A</b> = vektor gizi target, <b>B</b> = vektor gizi makanan kandidat.
        </div>
        """, unsafe_allow_html=True)

        section("🍽️ Proporsi Waktu Makan")
        meals = [
            ("☀️ Sarapan", "25%", "#f5a623", "Target kalori × 0.25"),
            ("🌤️ Makan Siang", "40%", "#2d9e5f", "Target kalori × 0.40"),
            ("🌙 Makan Malam", "35%", "#4a6fa5", "Target kalori × 0.35"),
        ]
        mc = st.columns(3)
        for i, (label, pct, color, desc) in enumerate(meals):
            with mc[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-top:4px solid {color};">
                    <div class="metric-num" style="color:{color};">{pct}</div>
                    <div class="metric-label">{label}</div>
                    <div style="font-size:0.75rem;color:#888;margin-top:6px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        section("📊 Evaluasi Content-Based Filtering")
        st.markdown("""
        <div class="section-desc" style="margin-bottom:12px;">
            Evaluasi menggunakan <b>Nutritional Coverage Rate</b> — seberapa % target gizi harian
            yang berhasil dipenuhi oleh menu yang di-generate, dirata-rata dari 10 variasi menu per profil.
        </div>
        """, unsafe_allow_html=True)

        profiles_eval = [
            ("🧒 SD Kelas 1–3", 100.0, "#2d9e5f"),
            ("👦 SD Kelas 4–6", 100.0, "#2d9e5f"),
            ("🎓 SMP/SMA", 99.9, "#2d9e5f"),
            ("🤱 Ibu Hamil/Menyusui", 98.6, "#f39c12"),
        ]
        for label, score, color in profiles_eval:
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;
                            font-weight:700;color:#1d2b22;margin-bottom:6px;">
                    <span>{label}</span>
                    <span style="color:{color};">{score}%</span>
                </div>
                <div style="background:#e8f5ee;border-radius:999px;height:10px;overflow:hidden;">
                    <div style="background:{color};width:{score}%;height:10px;border-radius:999px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-top:16px;border-top:4px solid #2d9e5f;">
            <div class="metric-num">99.6%</div>
            <div class="metric-unit">overall</div>
            <div class="metric-label">✅ Rata-rata Coverage Semua Profil</div>
        </div>
        """, unsafe_allow_html=True)

        section("📋 Ringkasan Sistem")
        features = [
            ("✅", "Metode", "Cosine Similarity"),
            ("✅", "Profil tersedia", "4 profil pengguna"),
            ("✅", "Waktu makan", "Sarapan, Siang, Malam"),
            ("✅", "Variasi menu", "Tidak ada makanan sama dalam 1 hari"),
            ("✅", "Menu mingguan", "Generate 7 hari sekaligus"),
        ]
        for icon, key, val in features:
            st.markdown(f"""
            <div style="display:flex;padding:7px 0;border-bottom:1px solid #f0f7f2;font-size:0.85rem;">
                <span style="margin-right:8px;">{icon}</span>
                <span style="color:#607566;flex:1;">{key}</span>
                <b style="color:#1d2b22;">{val}</b>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: DEMO
# ═══════════════════════════════════════════
elif page == "🍽️  Step 5: Demo Rekomendasi":
    hero("🍽️ Demo Rekomendasi Menu", "Generate menu bergizi harian & mingguan untuk berbagai profil", "🚀 Step 5 dari 5")

    if not model_loaded:
        st.error("⚠️ File `recommender.pkl` tidak ditemukan! Pastikan file ada di folder yang sama dengan `app.py`.")
        st.stop()

    # FORM INPUT
    st.markdown("### 👤 Pilih Profil Pengguna")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        profil_options = {
            '🧒 Anak SD Kelas 1–3  (Usia 7–9 tahun)'   : 'SD Kelas 1-3',
            '👦 Anak SD Kelas 4–6  (Usia 10–12 tahun)' : 'SD Kelas 4-6',
            '🎓 Siswa SMP/SMA       (Usia 13–18 tahun)' : 'SMP/SMA',
            '🤱 Ibu Hamil/Menyusui'                      : 'Ibu Hamil/Menyusui',
        }
        pilihan_display = st.selectbox("Siapa yang akan makan?", options=list(profil_options.keys()))
        profil_key = profil_options[pilihan_display]

    with col2:
        jumlah_hari = st.selectbox(
            "Mau menu untuk berapa hari?",
            options=[1, 3, 7],
            index=2,
            format_func=lambda x: f"{x} hari" if x > 1 else "1 hari (hari ini)"
        )

    with col3:
        target = PROFIL_GIZI[profil_key]
        st.markdown(f"""
        <div class="profil-card">
            <h3>🎯 Target Harian</h3>
            <div style="font-size:0.85rem; color:#555; line-height:2.2">
                🔥 <b>{target['kalori']}</b> kkal<br>
                💪 <b>{target['protein']}g</b> protein<br>
                💧 <b>{target['lemak']}g</b> lemak<br>
                🌾 <b>{target['karbo']}g</b> karbo
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🍽️ Generate Menu Sekarang!", use_container_width=True)

    if generate_btn:
        with st.spinner("Sedang menyusun menu terbaik untukmu..."):
            menu_list = generate_menu_mingguan(profil_key, jumlah_hari)

        HARI_NAMES = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        WAKTU_COLOR = {
            'Sarapan'    : 'linear-gradient(90deg,#e67e22,#f5a623)',
            'Makan Siang': 'linear-gradient(90deg,#1a6b3c,#2d9e5f)',
            'Makan Malam': 'linear-gradient(90deg,#2c3e7a,#4a6fa5)',
        }
        WAKTU_ICON = {
            'Sarapan'    : '☀️',
            'Makan Siang': '🌤️',
            'Makan Malam': '🌙',
        }

        st.markdown("---")
        st.markdown("### 📅 Hasil Rekomendasi Menu")

        for idx, menu in enumerate(menu_list):
            hari_label = HARI_NAMES[idx] if jumlah_hari > 1 else "Hari Ini"
            t   = menu['total']
            tgt = menu['target']
            pct_kal  = min(t['kalori']  / tgt['kalori']  * 100, 100)
            pct_prot = min(t['protein'] / tgt['protein'] * 100, 100)
            pct_lem  = min(t['lemak']   / tgt['lemak']   * 100, 100)
            pct_karb = min(t['karbo']   / tgt['karbo']   * 100, 100)

            with st.expander(f"{hari_label}  —  {t['kalori']:.0f} kkal  |  {t['protein']:.1f}g protein", expanded=(idx == 0)):
                col_menu, col_gizi = st.columns([3, 2])

                with col_menu:
                    for waktu, isi in menu['menu'].items():
                        g        = isi['gizi']
                        icon     = WAKTU_ICON.get(waktu, '🍴')
                        hdr_grad = WAKTU_COLOR.get(waktu, 'linear-gradient(90deg,#1a6b3c,#2d9e5f)')
                        karbo_s  = _h.escape(str(isi['karbo']))
                        prot_s   = _h.escape(str(isi['protein']))
                        sayur_s  = _h.escape(str(isi['sayur']))

                        st.markdown(f"""
                        <div class="meal-card">
                            <div class="meal-card-header" style="background:{hdr_grad};">
                                {icon}&nbsp;&nbsp;{waktu}
                            </div>
                            <div class="meal-card-body">
                                <div class="meal-col">
                                    <div class="meal-col-label">🌾 Karbohidrat</div>
                                    <div class="meal-col-name">{karbo_s}</div>
                                </div>
                                <div class="meal-col">
                                    <div class="meal-col-label">🍗 Protein</div>
                                    <div class="meal-col-name">{prot_s}</div>
                                </div>
                                <div class="meal-col">
                                    <div class="meal-col-label">🥬 Sayuran</div>
                                    <div class="meal-col-name">{sayur_s}</div>
                                </div>
                            </div>
                            <div class="meal-gizi-bar">
                                🔥 {g['kalori']:.0f} kkal
                                &nbsp;•&nbsp; 💪 Protein {g['protein']:.1f}g
                                &nbsp;•&nbsp; 💧 Lemak {g['lemak']:.1f}g
                                &nbsp;•&nbsp; 🌾 Karbo {g['karbo']:.1f}g
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_gizi:
                    st.markdown("""
                    <div class="gizi-panel">
                        <div class="gizi-panel-title">📊 Pemenuhan Gizi Harian</div>
                    """, unsafe_allow_html=True)

                    def gizi_bar(label, nilai, target_val, satuan, warna):
                        pct   = min(nilai / target_val * 100, 100)
                        color = warna if pct >= 70 else ("#f39c12" if pct >= 40 else "#e74c3c")
                        st.markdown(f"""
                        <div style="margin-bottom:14px;">
                            <div style="display:flex;justify-content:space-between;
                                        font-size:0.8rem;color:#607566;margin-bottom:6px;font-weight:700;">
                                <span>{label}</span>
                                <span style="color:#1d2b22;">{nilai:.1f}&thinsp;/&thinsp;{target_val} {satuan}
                                    <b style="color:{color};"> {pct:.0f}%</b>
                                </span>
                            </div>
                            <div style="background:#e0f0e6;border-radius:999px;height:9px;overflow:hidden;">
                                <div style="background:{color};width:{pct}%;height:9px;border-radius:999px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    gizi_bar('🔥 Kalori',  t['kalori'],  tgt['kalori'],  "kkal", "#2d9e5f")
                    gizi_bar('💪 Protein', t['protein'], tgt['protein'], "g",    "#3498db")
                    gizi_bar('💧 Lemak',   t['lemak'],   tgt['lemak'],   "g",    "#e67e22")
                    gizi_bar('🌾 Karbo',   t['karbo'],   tgt['karbo'],   "g",    "#9b59b6")

                    st.markdown("</div>", unsafe_allow_html=True)

                    avg_pct = (pct_kal + pct_prot + pct_lem + pct_karb) / 4
                    if avg_pct >= 75:
                        st.success("✅ Gizi harian terpenuhi dengan baik!")
                    elif avg_pct >= 50:
                        st.warning("⚠️ Gizi cukup, pertimbangkan tambahan camilan sehat.")
                    else:
                        st.error("❌ Gizi kurang tercukupi, tambahkan porsi makanan.")

        # RINGKASAN
        if jumlah_hari > 1:
            st.markdown("---")
            st.markdown("### 📈 Rata-rata Gizi Harian")

            avg_kal  = sum(m['total']['kalori']  for m in menu_list) / len(menu_list)
            avg_prot = sum(m['total']['protein'] for m in menu_list) / len(menu_list)
            avg_lem  = sum(m['total']['lemak']   for m in menu_list) / len(menu_list)
            avg_karb = sum(m['total']['karbo']   for m in menu_list) / len(menu_list)

            c1, c2, c3, c4 = st.columns(4)
            for col, label, nilai, satuan in [
                (c1, '🔥 Kalori',  avg_kal,  "kkal"),
                (c2, '💪 Protein', avg_prot, "gram"),
                (c3, '💧 Lemak',   avg_lem,  "gram"),
                (c4, '🌾 Karbo',   avg_karb, "gram"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number">{nilai:.0f}</div>
                        <div class="stat-unit">{satuan}/hari</div>
                        <div class="stat-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    🍽️ <b>PlatGizi</b> – Smart Menu Planner MBG &nbsp;|&nbsp;
    Machine Learning Project &nbsp;|&nbsp;
    Binus University LC01 &nbsp;|&nbsp;
    SDG 2 🌱 & SDG 3 💚
</div>
""", unsafe_allow_html=True)