import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PlatGizi – Smart Menu Planner MBG",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Playfair+Display:wght@700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}
.main { background-color: #f8fdf4; }

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
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin-top: 8px;
}
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
.profil-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    border: 2px solid #e8f5ee;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 24px;
}
.profil-card h3 { color: #1a6b3c; font-size: 1.2rem; margin-bottom: 4px; }
.stat-box {
    background: white;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 2px solid #e8f5ee;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stat-number { font-size: 1.8rem; font-weight: 900; color: #1a6b3c; line-height: 1; }
.stat-unit   { font-size: 0.8rem; color: #888; margin-top: 2px; }
.stat-label  { font-size: 0.85rem; color: #555; font-weight: 600; margin-top: 4px; }
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.8rem;
    margin-top: 8px;
    padding: 0px;
    border-top: 1px solid #e8f5ee;
}
/* Remove Streamlit's default bottom dead-space */
.main .block-container {
    padding-bottom: 1rem !important;
}
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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── MEAL CARDS ── */
.meal-card {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e0f0e6;
    overflow: hidden;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(26,107,60,0.07);
    font-family: 'Nunito', sans-serif;
}
.meal-card-header {
    padding: 10px 16px;
    font-weight: 800;
    font-size: 0.95rem;
    color: white;
    letter-spacing: 0.3px;
}
.meal-card-body {
    display: flex;
}
.meal-col {
    flex: 1;
    padding: 12px 14px;
    border-right: 1px solid #f0f7f2;
}
.meal-col:last-child { border-right: none; }
.meal-col-label {
    font-size: 0.68rem;
    font-weight: 800;
    color: #607566;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-bottom: 5px;
}
.meal-col-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #1d2b22;
    line-height: 1.4;
}
.meal-gizi-bar {
    background: #f4fbf6;
    padding: 7px 16px;
    font-size: 0.77rem;
    color: #607566;
    border-top: 1px solid #e8f5ee;
    font-weight: 600;
    font-family: 'Nunito', sans-serif;
}

/* ── GIZI PANEL ── */
.gizi-panel {
    background: #f8fdf4;
    border-radius: 14px;
    padding: 16px 18px;
    border: 1px solid #e8f5ee;
    height: fit-content;
}
.gizi-panel-title {
    font-size: 0.92rem;
    font-weight: 800;
    color: #1a6b3c;
    margin-bottom: 14px;
    font-family: 'Nunito', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_recommender():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'recommender.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    rec          = load_recommender()
    nutrition_df = rec['nutrition_df']
    resep_df     = rec['resep_df']
    scaler       = rec['scaler']
    kmeans       = rec['kmeans']
    PROFIL_GIZI  = rec['profil_gizi']
    model_loaded = True
except:
    model_loaded = False


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
    subset               = subset.copy()
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

        # Filter kalori agar tidak melebihi target per waktu makan
        karbo_cand = karbo_cand[karbo_cand['calories'] <= tk * 0.7] if len(karbo_cand) > 0 else karbo_cand
        prot_cand  = prot_cand[prot_cand['calories']   <= tk * 0.6] if len(prot_cand)  > 0 else prot_cand
        sayur_cand = sayur_cand[sayur_cand['calories'] <= tk * 0.3] if len(sayur_cand) > 0 else sayur_cand

        # Fallback jika filter terlalu ketat
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
            'kalori' : (karbo['calories']     if karbo is not None else 0) +
                       (prot['calories']      if prot  is not None else 0) +
                       (sayur['calories']     if sayur is not None else 0),
            'protein': (karbo['proteins']     if karbo is not None else 0) +
                       (prot['proteins']      if prot  is not None else 0) +
                       (sayur['proteins']     if sayur is not None else 0),
            'lemak'  : (karbo['fat']          if karbo is not None else 0) +
                       (prot['fat']           if prot  is not None else 0) +
                       (sayur['fat']          if sayur is not None else 0),
            'karbo'  : (karbo['carbohydrate'] if karbo is not None else 0) +
                       (prot['carbohydrate']  if prot  is not None else 0) +
                       (sayur['carbohydrate'] if sayur is not None else 0),
        }
        menu[waktu] = {
            'karbo'  : karbo['name'] if karbo is not None else '-',
            'protein': prot['name']  if prot  is not None else '-',
            'sayur'  : sayur['name'] if sayur is not None else '-',
            'gizi'   : gizi,
        }
        for k in total:
            total[k] += gizi[k]

    return {
        'profil': profil,
        'target': target,
        'menu'  : menu,
        'total' : {k: round(v, 1) for k, v in total.items()}
    }


def generate_menu_mingguan(profil, jumlah_hari=7):
    return [
        {**generate_menu_harian(profil, seed=hari * 13), 'hari': hari + 1}
        for hari in range(jumlah_hari)
    ]


# ─────────────────────────────────────────
# TAMPILAN UTAMA
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🍽️ PlatGizi</div>
    <div class="hero-subtitle">Smart Menu Planner untuk Program Makan Bergizi Gratis (MBG)</div>
    <div class="hero-badge">🤖 Powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

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
    pilihan_display = st.selectbox(
        "Siapa yang akan makan?",
        options=list(profil_options.keys()),
        help="Pilih profil sesuai kebutuhan. Setiap profil memiliki target kalori & gizi yang berbeda."
    )
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
        <h3><i class="fa-solid fa-bullseye" style="color:#1a6b3c"></i> Target Harian</h3>
        <div style="font-size:0.85rem; color:#555; line-height:2.2">
            <i class="fa-solid fa-fire" style="color:#e74c3c"></i> <b>{target['kalori']}</b> kkal<br>
            <i class="fa-solid fa-dumbbell" style="color:#3498db"></i> <b>{target['protein']}g</b> protein<br>
            <i class="fa-solid fa-droplet" style="color:#f39c12"></i> <b>{target['lemak']}g</b> lemak<br>
            <i class="fa-solid fa-wheat-awn" style="color:#9b59b6"></i> <b>{target['karbo']}g</b> karbo
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("🍽️ Generate Menu Sekarang!", use_container_width=True)

# OUTPUT MENU
if generate_btn:
    with st.spinner("Sedang menyusun menu terbaik untukmu..."):
        menu_list = generate_menu_mingguan(profil_key, jumlah_hari)

    HARI_NAMES = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    WAKTU_ICON = {
        'Sarapan'    : '<i class="fa-solid fa-sun" style="color:#f5a623"></i>',
        'Makan Siang': '<i class="fa-solid fa-cloud-sun" style="color:#e67e22"></i>',
        'Makan Malam': '<i class="fa-solid fa-moon" style="color:#c8a8e9"></i>',
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
                import html as _h
                WAKTU_COLOR = {
                    'Sarapan'    : 'linear-gradient(90deg,#e67e22,#f5a623)',
                    'Makan Siang': 'linear-gradient(90deg,#1a6b3c,#2d9e5f)',
                    'Makan Malam': 'linear-gradient(90deg,#2c3e7a,#4a6fa5)',
                }
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
                                <div class="meal-col-label"><i class="fa-solid fa-wheat-awn" style="color:#9b59b6"></i> Karbohidrat</div>
                                <div class="meal-col-name">{karbo_s}</div>
                            </div>
                            <div class="meal-col">
                                <div class="meal-col-label"><i class="fa-solid fa-drumstick-bite" style="color:#e67e22"></i> Protein</div>
                                <div class="meal-col-name">{prot_s}</div>
                            </div>
                            <div class="meal-col">
                                <div class="meal-col-label"><i class="fa-solid fa-leaf" style="color:#27ae60"></i> Sayuran</div>
                                <div class="meal-col-name">{sayur_s}</div>
                            </div>
                        </div>
                        <div class="meal-gizi-bar">
                            <i class="fa-solid fa-fire" style="color:#e74c3c"></i> {g['kalori']:.0f} kkal
                            &nbsp;•&nbsp; <i class="fa-solid fa-dumbbell" style="color:#3498db"></i> Protein {g['protein']:.1f}g
                            &nbsp;•&nbsp; <i class="fa-solid fa-droplet" style="color:#f39c12"></i> Lemak {g['lemak']:.1f}g
                            &nbsp;•&nbsp; <i class="fa-solid fa-wheat-awn" style="color:#9b59b6"></i> Karbo {g['karbo']:.1f}g
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_gizi:
                st.markdown("""
                <div class="gizi-panel">
                    <div class="gizi-panel-title"><i class="fa-solid fa-chart-bar" style="color:#1a6b3c"></i> Pemenuhan Gizi Harian</div>
                """, unsafe_allow_html=True)

                def gizi_bar(label, nilai, target_val, satuan, warna):
                    pct   = min(nilai / target_val * 100, 100)
                    color = warna if pct >= 70 else ("#f39c12" if pct >= 40 else "#e74c3c")
                    st.markdown(f"""
                    <div style="margin-bottom:14px;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:0.8rem;color:#607566;
                                    margin-bottom:6px;font-weight:700;
                                    font-family:'Nunito',sans-serif;">
                            <span>{label}</span>
                            <span style="color:#1d2b22;">{nilai:.1f}&thinsp;/&thinsp;{target_val} {satuan}
                                <b style="color:{color};"> {pct:.0f}%</b>
                            </span>
                        </div>
                        <div style="background:#e0f0e6;border-radius:999px;height:9px;overflow:hidden;">
                            <div style="background:{color};width:{pct}%;height:9px;
                                        border-radius:999px;transition:width 0.6s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                gizi_bar('<i class="fa-solid fa-fire" style="color:#e74c3c"></i> Kalori',  t['kalori'],  tgt['kalori'],  "kkal", "#2d9e5f")
                gizi_bar('<i class="fa-solid fa-dumbbell" style="color:#3498db"></i> Protein', t['protein'], tgt['protein'], "g",    "#3498db")
                gizi_bar('<i class="fa-solid fa-droplet" style="color:#f39c12"></i> Lemak',   t['lemak'],   tgt['lemak'],   "g",    "#e67e22")
                gizi_bar('<i class="fa-solid fa-wheat-awn" style="color:#9b59b6"></i> Karbo',   t['karbo'],   tgt['karbo'],   "g",    "#9b59b6")

                st.markdown("</div>", unsafe_allow_html=True)

                avg_pct = (pct_kal + pct_prot + pct_lem + pct_karb) / 4
                if avg_pct >= 75:
                    st.success("✅ Gizi harian terpenuhi dengan baik!")
                elif avg_pct >= 50:
                    st.warning("⚠️ Gizi cukup, pertimbangkan tambahan camilan sehat.")
                else:
                    st.error("❌ Gizi kurang tercukupi, tambahkan porsi makanan.")


    # RINGKASAN MINGGUAN
    if jumlah_hari > 1:
        st.markdown("---")
        st.markdown("### 📈 Rata-rata Gizi Harian (Selama Periode)")

        avg_kal  = sum(m['total']['kalori']  for m in menu_list) / len(menu_list)
        avg_prot = sum(m['total']['protein'] for m in menu_list) / len(menu_list)
        avg_lem  = sum(m['total']['lemak']   for m in menu_list) / len(menu_list)
        avg_karb = sum(m['total']['karbo']   for m in menu_list) / len(menu_list)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, nilai, satuan in [
            (c1, '<i class="fa-solid fa-fire" style="color:#e74c3c"></i> Kalori',  avg_kal,  "kkal"),
            (c2, '<i class="fa-solid fa-dumbbell" style="color:#3498db"></i> Protein', avg_prot, "gram"),
            (c3, '<i class="fa-solid fa-droplet" style="color:#f39c12"></i> Lemak',   avg_lem,  "gram"),
            (c4, '<i class="fa-solid fa-wheat-awn" style="color:#9b59b6"></i> Karbo',   avg_karb, "gram"),
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
    SDG 2🌱
</div>
""", unsafe_allow_html=True)
