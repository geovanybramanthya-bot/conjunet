"""
CONJUNET Dashboard v5 - Redesign Total
=======================================
Sistem desain: Teal + Cream, medis & humanis, mobile-first.
Navigasi: Beranda | Skrining | Edukasi | Riwayat | Statistik
Konten kontekstual: Panduan Foto (dalam Skrining), Tentang (dalam Beranda)

Ensemble 3 model (seed 42,123,456), threshold 0.44 (dikalibrasi dari val set).

BUG HISTORY yang dijaga jangan terulang:
[BUG-1] rescale=1./255 -> model collapse. SOLUSI: tanpa rescale.
[BUG-2] cache tanpa key -> model basi. SOLUSI: cache key = signature file.
[BUG-3] confidence terbalik di DB. SOLUSI: conf_simpan per zona.
[BUG-4] threshold default 0.5. SOLUSI: 0.44 dari val.
[BUG-5] threshold dari test set. SOLUSI: sudah dari val set.
"""

import streamlit as st
import sqlite3
import hashlib
import datetime
import numpy as np
import os
from pathlib import Path

st.set_page_config(
    page_title="CONJUNET - Skrining Anemia",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

MODEL_PATHS = [
    "conjunet_seed42.keras",
    "conjunet_seed123.keras",
    "conjunet_seed456.keras",
]
DB_PATH   = "conjunet_pasien.db"
IMG_SIZE  = (224, 224)
THRESHOLD = 0.44

# HuggingFace: download model kalau tidak ada di lokal (untuk cloud deployment)
HF_BASE = "https://huggingface.co/WrathOfAzura/conjunet-models/resolve/main"

def ensure_models():
    """Download model dari HuggingFace kalau belum ada di lokal."""
    import urllib.request
    all_ok = True
    for fname in MODEL_PATHS:
        if not Path(fname).exists():
            url = f"{HF_BASE}/{fname}"
            try:
                with st.spinner(f"Mengunduh model {fname}... (sekali saja, ~40MB)"):
                    urllib.request.urlretrieve(url, fname)
            except Exception as e:
                st.error(f"Gagal mengunduh {fname}: {e}")
                all_ok = False
    return all_ok

# ── SISTEM DESAIN (CSS) ───────────────────────────
# Palet: Teal (#0E7C7B primary, #14A098 mid, #41B3A3 soft) + Cream (#FBF7F0)
# Aksen hangat: #E8A87C (coral lembut untuk highlight)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --teal-deep: #0E7C7B;
        --teal-mid: #14A098;
        --teal-soft: #41B3A3;
        --teal-pale: #D9EDE9;
        --cream: #FBF7F0;
        --cream-card: #FFFFFF;
        --ink: #1C3835;
        --ink-soft: #4A615E;
        --coral: #E8A87C;
    }

    .stApp { background: var(--cream); }
    .block-container { padding-top: 1rem; padding-bottom: 5rem; max-width: 760px; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1,h2,h3,h4,h5,h6 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--ink) !important; }
    .stMarkdown p { color: var(--ink-soft); }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; height: 0; }

    /* ── HERO dengan gradasi teal ── */
    .hero {
        background: linear-gradient(135deg, #0E7C7B 0%, #14A098 55%, #41B3A3 100%);
        border-radius: 28px; padding: 2.4rem 2rem; margin-bottom: 1.2rem;
        position: relative; overflow: hidden;
        box-shadow: 0 12px 32px rgba(14,124,123,0.18);
    }
    /* Hero dengan foto background (overlay gradasi teal di atas foto) */
    .hero-photo {
        position: relative; overflow: hidden;
        border-radius: 28px; padding: 2.4rem 2rem; margin-bottom: 1.2rem;
        box-shadow: 0 12px 32px rgba(14,124,123,0.18);
        background-size: cover; background-position: center;
    }
    .hero-photo::before {
        content: ''; position: absolute; inset: 0;
        background: linear-gradient(135deg, rgba(14,124,123,0.92) 0%, rgba(20,160,152,0.82) 55%, rgba(65,179,163,0.78) 100%);
        z-index: 0;
    }
    .hero-photo > * { position: relative; z-index: 1; }
    .hero-login-bg { background-image: url('https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200&q=80'); }
    .hero-home-bg { background-image: url('https://images.unsplash.com/photo-1542736667-069246bdbc6d?w=1200&q=80'); }
    .hero-edu-bg { background-image: url('https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=1200&q=80'); }
    .hero-stat-bg { background-image: url('https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?w=1200&q=80'); }
    .hero-screen-bg { background-image: url('https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=1200&q=80'); }
    .hero-hist-bg { background-image: url('https://images.unsplash.com/photo-1554734867-bf3c00a49371?w=1200&q=80'); }
    .hero::after {
        content: ''; position: absolute; top: -40px; right: -40px;
        width: 160px; height: 160px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }
    .hero::before {
        content: ''; position: absolute; bottom: -60px; left: -30px;
        width: 140px; height: 140px; border-radius: 50%;
        background: rgba(255,255,255,0.06);
    }
    .hero-logo { font-size: 0.78rem; font-weight: 700; letter-spacing: 3px; color: rgba(255,255,255,0.85); margin-bottom: 0.7rem; }
    .hero-title { font-family: 'Plus Jakarta Sans'; font-size: 2.1rem; font-weight: 800; line-height: 1.18; color: #FFFFFF !important; margin: 0 0 0.6rem 0; position: relative; z-index: 1; }
    .hero-sub { font-size: 0.98rem; color: rgba(255,255,255,0.92); margin: 0; line-height: 1.55; position: relative; z-index: 1; }

    .stat-row { display: flex; gap: 0.8rem; margin-top: 1.5rem; position: relative; z-index: 1; }
    .stat-chip { background: rgba(255,255,255,0.16); border-radius: 14px; padding: 0.7rem 0.9rem; flex: 1; backdrop-filter: blur(4px); }
    .stat-num { font-family: 'Plus Jakarta Sans'; font-size: 1.35rem; font-weight: 800; color: #FFFFFF; line-height: 1; }
    .stat-label { font-size: 0.72rem; color: rgba(255,255,255,0.82); margin-top: 0.25rem; }

    /* ── Section heading ── */
    .sec-head { font-family: 'Plus Jakarta Sans'; font-size: 1.2rem; font-weight: 700; color: var(--ink); margin: 1.6rem 0 0.4rem 0; }
    .sec-caption { font-size: 0.9rem; color: var(--ink-soft); margin-bottom: 0.7rem; }

    /* ── Kartu umum ── */
    .card {
        background: var(--cream-card); border-radius: 20px; padding: 1.4rem 1.5rem;
        margin: 0.7rem 0; box-shadow: 0 4px 18px rgba(28,56,53,0.05);
        border: 1px solid rgba(14,124,123,0.06);
    }
    .card-teal {
        background: linear-gradient(135deg, #D9EDE9, #EAF5F2);
        border-radius: 20px; padding: 1.4rem 1.5rem; margin: 0.7rem 0;
        border: 1px solid rgba(14,124,123,0.12);
    }

    /* ── Tombol ── */
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(135deg, #0E7C7B, #14A098) !important;
        color: #FFFFFF !important; border: none !important; border-radius: 14px !important;
        font-family: 'Plus Jakarta Sans' !important; font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important; transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(14,124,123,0.22) !important;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(14,124,123,0.3) !important;
    }

    /* ── Input ── */
    .stTextInput input, .stNumberInput input {
        background: #FFFFFF !important; color: var(--ink) !important;
        border: 2px solid var(--teal-pale) !important; border-radius: 12px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus { border-color: var(--teal-mid) !important; box-shadow: none !important; }
    .stTextInput label, .stNumberInput label { color: var(--ink) !important; font-weight: 600 !important; }
    .stNumberInput button { background: #FFFFFF !important; color: var(--ink) !important; border: 2px solid var(--teal-pale) !important; }
    [data-testid="stFileUploaderDropzone"] { background: #FFFFFF !important; border: 2px dashed var(--teal-soft) !important; border-radius: 14px !important; }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink-soft) !important; }
</style>
""", unsafe_allow_html=True)

# ── CSS lanjutan: hasil, navigasi, komponen khusus ──
st.markdown("""
<style>
    /* Kartu hasil skrining */
    .hasil-card { padding: 1.7rem; border-radius: 20px; margin: 1rem 0; }
    .card-anemia { background: linear-gradient(135deg, #FCECEC, #FBE4E4); border: 2px solid #E07A7A; }
    .card-normal { background: linear-gradient(135deg, #E3F3EC, #D9EDE9); border: 2px solid #41B3A3; }
    .card-ragu   { background: linear-gradient(135deg, #FDF2E3, #FBEAD2); border: 2px solid #E8A87C; }
    .hasil-judul { font-family: 'Plus Jakarta Sans'; font-size: 1.35rem; font-weight: 700; color: var(--ink); margin: 0; }
    .hasil-pct { font-family: 'Plus Jakarta Sans'; font-size: 2rem; font-weight: 800; color: var(--ink); margin: 0.3rem 0; }
    .hasil-pesan { color: var(--ink-soft); margin: 0; line-height: 1.5; }

    /* Panel probabilitas */
    .prob-panel { background: #FFFFFF; border-radius: 18px; padding: 1.3rem 1.5rem; margin: 0.8rem 0; border: 1px solid rgba(14,124,123,0.1); }
    .prob-title { font-family: 'Plus Jakarta Sans'; font-weight: 700; font-size: 0.98rem; color: var(--ink); margin-bottom: 0.9rem; }
    .prob-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; }
    .prob-label { font-size: 0.92rem; color: var(--ink-soft); font-weight: 500; }
    .prob-val { font-family: 'Plus Jakarta Sans'; font-weight: 700; font-size: 0.98rem; color: var(--ink); }
    .prob-bar-bg { width: 100%; height: 11px; background: #EEF3F1; border-radius: 20px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 20px; transition: width 0.7s cubic-bezier(0.4,0,0.2,1); }
    .prob-bar-fill.anemia { background: linear-gradient(90deg, #E07A7A, #EF9A9A); }
    .prob-bar-fill.normal { background: linear-gradient(90deg, #0E7C7B, #41B3A3); }

    /* Info & disclaimer */
    .info-box { background: var(--teal-pale); padding: 1rem 1.2rem; border-radius: 14px; border-left: 4px solid var(--teal-mid); margin: 0.6rem 0; color: var(--ink); font-size: 0.92rem; }
    .disclaimer { font-size: 0.84rem; color: var(--ink-soft); background: #F2F0E9; padding: 1rem 1.2rem; border-radius: 14px; margin-top: 1rem; line-height: 1.6; }

    /* Item daftar (riwayat, langkah, dll) */
    .list-item { background: #FFFFFF; border-radius: 12px; padding: 0.8rem 1rem; margin: 0.4rem 0; border-left: 3px solid var(--teal-soft); font-size: 0.92rem; color: var(--ink-soft); }

    /* Kartu fitur di beranda */
    .feature-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin: 0.8rem 0; }
    .feature-card { background: #FFFFFF; border-radius: 18px; padding: 1.2rem; border: 1px solid rgba(14,124,123,0.08); box-shadow: 0 3px 12px rgba(28,56,53,0.04); }
    .feature-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .feature-title { font-family: 'Plus Jakarta Sans'; font-weight: 700; font-size: 1rem; color: var(--ink); margin-bottom: 0.3rem; }
    .feature-desc { font-size: 0.85rem; color: var(--ink-soft); line-height: 1.45; }

    /* Langkah panduan foto */
    .step-card { display: flex; gap: 1rem; align-items: flex-start; background: #FFFFFF; border-radius: 16px; padding: 1.1rem 1.3rem; margin: 0.6rem 0; border: 1px solid rgba(14,124,123,0.08); }
    .step-num { background: linear-gradient(135deg, #0E7C7B, #41B3A3); color: #FFFFFF; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'Plus Jakarta Sans'; font-weight: 700; font-size: 1.05rem; }
    .step-body { flex: 1; }
    .step-title { font-family: 'Plus Jakarta Sans'; font-weight: 700; color: var(--ink); font-size: 0.98rem; margin-bottom: 0.2rem; }
    .step-desc { font-size: 0.88rem; color: var(--ink-soft); line-height: 1.45; }

    /* Edukasi: kartu makanan */
    .food-card { background: #FFFFFF; border-radius: 16px; padding: 1rem; text-align: center; border: 1px solid rgba(14,124,123,0.08); }
    .food-emoji { font-size: 2.2rem; }
    .food-name { font-family: 'Plus Jakarta Sans'; font-weight: 600; font-size: 0.9rem; color: var(--ink); margin-top: 0.4rem; }

    /* Zona statistik */
    .zona-card { border-radius: 16px; padding: 1.2rem 1.4rem; margin: 0.6rem 0; color: #FFFFFF; }
    .zona-paradoks { background: linear-gradient(135deg, #E8A87C, #EBB68F); }
    .zona-kritis { background: linear-gradient(135deg, #E07A7A, #E89292); }
    .zona-aman { background: linear-gradient(135deg, #0E7C7B, #41B3A3); }
    .zona-nama { font-family: 'Plus Jakarta Sans'; font-weight: 700; font-size: 1.1rem; }
    .zona-detail { font-size: 0.88rem; opacity: 0.95; margin-top: 0.3rem; line-height: 1.5; }

    @keyframes fadeUp { from {opacity:0; transform:translateY(12px);} to {opacity:1; transform:translateY(0);} }
    .hero, .card, .card-teal, .hasil-card, .prob-panel, .feature-card, .step-card { animation: fadeUp 0.5s ease both; }

    /* MOBILE */
    @media (max-width: 640px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        .hero { padding: 1.8rem 1.3rem; border-radius: 22px; }
        .hero-title { font-size: 1.65rem; }
        .stat-row { gap: 0.5rem; }
        .stat-num { font-size: 1.1rem; }
        .stat-label { font-size: 0.65rem; }
        .feature-grid { grid-template-columns: 1fr; }
        .hasil-pct { font-size: 1.7rem; }
    }
</style>
""", unsafe_allow_html=True)

# ── DATABASE ──────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS pemeriksaan (
        id INTEGER PRIMARY KEY AUTOINCREMENT, id_pasien TEXT NOT NULL,
        nama_samaran TEXT, usia INTEGER, tanggal TEXT NOT NULL,
        hasil TEXT NOT NULL, confidence REAL NOT NULL, petugas TEXT)""")
    conn.commit(); conn.close()

def simpan_pemeriksaan(idp, nama, usia, hasil, conf, petugas):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""INSERT INTO pemeriksaan
        (id_pasien, nama_samaran, usia, tanggal, hasil, confidence, petugas)
        VALUES (?,?,?,?,?,?,?)""",
        (idp, nama, usia, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
         hasil, conf, petugas))
    conn.commit(); conn.close()

def ambil_riwayat(idp):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT tanggal, hasil, confidence FROM pemeriksaan WHERE id_pasien=? ORDER BY tanggal DESC", (idp,))
    rows = c.fetchall(); conn.close(); return rows

def semua_riwayat():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT id_pasien, nama_samaran, tanggal, hasil, confidence FROM pemeriksaan ORDER BY tanggal DESC LIMIT 50")
    rows = c.fetchall(); conn.close(); return rows

# ── LOGIN ─────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
AKUN_DEMO = {
    "bidan_desa":     (hash_pw("demo123"), "Bidan Sari"),
    "kader_posyandu": (hash_pw("demo123"), "Kader Ani"),
}
def cek_login(u, p):
    if u in AKUN_DEMO and hash_pw(p) == AKUN_DEMO[u][0]:
        return AKUN_DEMO[u][1]
    return None

# ── MODEL ENSEMBLE ────────────────────────────────
def _signatures():
    parts = []
    for p in MODEL_PATHS:
        try:
            s = os.stat(p); parts.append(f"{p}:{s.st_size}:{s.st_mtime}")
        except OSError:
            parts.append(f"{p}:missing")
    return "|".join(parts)

@st.cache_resource
def muat_ensemble(sig):
    import tensorflow as tf
    models, missing = [], []
    for p in MODEL_PATHS:
        if not Path(p).exists():
            missing.append(p); continue
        models.append(tf.keras.models.load_model(p))
    if missing: return None, missing
    return models, []

def prediksi_ensemble(models, image_pil):
    # Tanpa rescale (FIX BUG-1). Rata-rata 3 model. Return P_Anemia.
    img = image_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    raws = [float(m.predict(arr, verbose=0)[0][0]) for m in models]
    return 1.0 - np.mean(raws)

def interpretasi(pa):
    if pa >= 0.60:
        return {"status":"anemia","judul":"Terdapat Indikasi Anemia","pct":round(pa*100),
            "pesan":"Hasil skrining menunjukkan tanda kepucatan konjungtiva yang mengarah pada kemungkinan anemia.",
            "rekomendasi":[
                "Disarankan melakukan pemeriksaan darah (cek Hb) di puskesmas atau fasilitas kesehatan terdekat untuk memastikan.",
                "Perbanyak makanan kaya zat besi: sayuran hijau, hati, daging merah, telur, dan kacang-kacangan.",
                "Konsumsi sumber vitamin C (jeruk, jambu, tomat) bersama makanan berzat besi untuk membantu penyerapan."],
            "card":"card-anemia","conf_simpan":pa}
    elif pa >= 0.44:
        return {"status":"ragu","judul":"Hasil Belum Pasti","pct":round(pa*100),
            "pesan":"Hasil berada di zona abu-abu. Tanda anemia belum cukup jelas untuk disimpulkan.",
            "rekomendasi":[
                "Perbanyak makanan bergizi kaya zat besi (sayuran hijau, hati, daging, telur, kacang).",
                "Lakukan pemeriksaan ulang dalam 1 sampai 2 minggu ke depan menggunakan aplikasi ini.",
                "Jika hasil ulang tetap menunjukkan indikasi anemia, barulah disarankan tes darah di puskesmas."],
            "card":"card-ragu","conf_simpan":pa}
    else:
        return {"status":"normal","judul":"Tidak Terdapat Indikasi Anemia","pct":round((1-pa)*100),
            "pesan":"Hasil tidak menunjukkan tanda kepucatan konjungtiva yang signifikan.",
            "rekomendasi":[
                "Pertahankan pola makan bergizi seimbang dengan prinsip Isi Piringku: separuh piring berisi sayur dan buah, seperempat lauk pauk (protein), dan seperempat makanan pokok (karbohidrat).",
                "Jaga asupan zat besi rutin dari sumber alami seperti sayuran hijau, daging, telur, dan kacang-kacangan, meski hasil saat ini normal. Pencegahan lebih baik daripada pemulihan.",
                "Untuk ibu hamil, tetap konsumsi Tablet Tambah Darah (TTD) sesuai anjuran bidan minimal 90 tablet selama kehamilan, karena kebutuhan zat besi meningkat tajam.",
                "Lakukan skrining ulang secara berkala, terutama menjelang dan selama kehamilan, untuk memastikan kondisi tetap terjaga."],
            "card":"card-normal","conf_simpan":1.0-pa}

init_db()
if "login" not in st.session_state:
    st.session_state.login = False
    st.session_state.petugas = None
if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

# ── HALAMAN LOGIN ─────────────────────────────────
def halaman_login():
    st.markdown("""
    <div class='hero-photo hero-login-bg'>
        <div class='hero-logo'>🩺 CONJUNET</div>
        <div class='hero-title'>Skrining Anemia<br>Tanpa Jarum, Tanpa Lab</div>
        <p class='hero-sub'>Deteksi dini anemia berbasis citra konjungtiva,
        dirancang untuk menjangkau hingga pelosok desa.</p>
        <div class='stat-row'>
            <div class='stat-chip'><div class='stat-num'>0,94</div><div class='stat-label'>AUC Model</div></div>
            <div class='stat-chip'><div class='stat-num'>86,7%</div><div class='stat-label'>Sensitivitas</div></div>
            <div class='stat-chip'><div class='stat-num'>&lt;5 dtk</div><div class='stat-label'>Per Skrining</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='info-box'>🔒 Aplikasi ini khusus untuk tenaga kesehatan dan "
                "kader posyandu. Data pasien Anda aman dan tersimpan secara privat.</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>Masuk ke Akun Anda</div>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Nama Pengguna")
        password = st.text_input("Kata Sandi", type="password")
        submit = st.form_submit_button("Masuk")
    if submit:
        nama = cek_login(username, password)
        if nama:
            st.session_state.login = True
            st.session_state.petugas = nama
            st.rerun()
        else:
            st.error("Nama pengguna atau kata sandi salah.")

    st.markdown("<div class='disclaimer'><b>Akun demo:</b> &nbsp; "
                "Pengguna <b>bidan_desa</b> atau <b>kader_posyandu</b> &nbsp;|&nbsp; "
                "Sandi <b>demo123</b></div>", unsafe_allow_html=True)

# ── NAVIGASI ──────────────────────────────────────
def navigasi():
    menus = ["Beranda", "Skrining", "Edukasi", "Riwayat", "Statistik"]
    cols = st.columns(len(menus))
    for i, m in enumerate(menus):
        with cols[i]:
            tipe = "primary" if st.session_state.menu == m else "secondary"
            if st.button(m, key=f"nav_{m}", use_container_width=True, type=tipe):
                st.session_state.menu = m
                st.rerun()

# ── HALAMAN: BERANDA ──────────────────────────────
def page_beranda():
    st.markdown(f"""
    <div class='hero-photo hero-home-bg'>
        <div class='hero-logo'>🩺 CONJUNET</div>
        <div class='hero-title'>Selamat datang,<br>{st.session_state.petugas}</div>
        <p class='hero-sub'>Mari bantu deteksi dini anemia di komunitas Anda.
        Pilih menu di bawah untuk memulai.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>Apa yang ingin Anda lakukan?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='feature-grid'>
        <div class='feature-card'>
            <div class='feature-icon'>🔬</div>
            <div class='feature-title'>Skrining Cepat</div>
            <div class='feature-desc'>Unggah foto konjungtiva, dapatkan hasil dalam hitungan detik.</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>📚</div>
            <div class='feature-title'>Edukasi Gizi</div>
            <div class='feature-desc'>Panduan makanan kaya zat besi untuk cegah anemia.</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>📋</div>
            <div class='feature-title'>Riwayat Pasien</div>
            <div class='feature-desc'>Pantau perkembangan hasil skrining antar kunjungan.</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>🗺️</div>
            <div class='feature-title'>Statistik Daerah</div>
            <div class='feature-desc'>Lihat peta zona kerentanan anemia di Indonesia.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>Mengapa Konjungtiva?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    Kepucatan pada konjungtiva (bagian dalam kelopak mata bawah) telah lama dikenal
    dalam dunia medis sebagai indikator visual kadar hemoglobin. CONJUNET memanfaatkan
    kecerdasan buatan untuk membaca tanda ini secara objektif, menjadikannya alat bantu
    skrining yang cepat, murah, dan tidak menyakitkan.
    </div>
    """, unsafe_allow_html=True)

    # TENTANG (kontekstual di beranda)
    st.markdown("<div class='sec-head'>Tentang CONJUNET</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card-teal'>
    CONJUNET dikembangkan sebagai sistem skrining anemia noninvasif untuk mendukung
    pemerataan layanan kesehatan di Indonesia, khususnya bagi ibu hamil di wilayah dengan
    akses laboratorium terbatas. Sistem ini menggabungkan model kecerdasan buatan
    EfficientNet-B0 dengan analisis sebaran kerentanan antardaerah. <br><br>
    <b>Penting:</b> CONJUNET adalah alat bantu skrining awal, bukan pengganti diagnosis medis.
    </div>
    """, unsafe_allow_html=True)

# ── HALAMAN: SKRINING (+ Panduan Foto kontekstual) ──
def page_skrining(models):
    from PIL import Image

    st.markdown("""
    <div class='hero-photo hero-screen-bg'>
        <div class='hero-logo'>🔬 SKRINING</div>
        <div class='hero-title'>Pemeriksaan Baru</div>
        <p class='hero-sub'>Ikuti panduan foto, isi data pasien, lalu unggah citra konjungtiva.</p>
    </div>
    """, unsafe_allow_html=True)

    # Panduan foto kontekstual (expander)
    with st.expander("📸 Lihat Panduan Cara Foto yang Benar"):
        st.markdown("""
        <div class='step-card'><div class='step-num'>1</div><div class='step-body'>
            <div class='step-title'>Tarik kelopak mata bawah</div>
            <div class='step-desc'>Minta pasien melihat ke atas, lalu tarik lembut kelopak mata bawah hingga konjungtiva (bagian dalam berwarna merah muda) terlihat jelas.</div>
        </div></div>
        <div class='step-card'><div class='step-num'>2</div><div class='step-body'>
            <div class='step-title'>Pencahayaan cukup</div>
            <div class='step-desc'>Gunakan cahaya alami atau ruangan terang. Hindari penggunaan flash karena dapat mengubah warna asli konjungtiva.</div>
        </div></div>
        <div class='step-card'><div class='step-num'>3</div><div class='step-body'>
            <div class='step-title'>Fokus dan dekat</div>
            <div class='step-desc'>Posisikan kamera 10-15 cm dari mata. Pastikan area konjungtiva tampak tajam dan tidak buram.</div>
        </div></div>
        <div class='step-card'><div class='step-num'>4</div><div class='step-body'>
            <div class='step-title'>Hindari bayangan</div>
            <div class='step-desc'>Pastikan tidak ada bayangan jari atau benda lain yang menutupi area konjungtiva saat memotret.</div>
        </div></div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sec-caption' style='margin-top:1rem;'><b>Contoh visual: benar vs salah</b></div>", unsafe_allow_html=True)
        col_benar, col_salah = st.columns(2)
        with col_benar:
            st.markdown("""
            <div style='background:#FFFFFF; border:2px solid #41B3A3; border-radius:16px; padding:1rem; text-align:center;'>
                <svg viewBox="0 0 120 80" width="100%" height="90">
                    <ellipse cx="60" cy="40" rx="50" ry="28" fill="#FCE4E0"/>
                    <ellipse cx="60" cy="40" rx="50" ry="28" fill="none" stroke="#41B3A3" stroke-width="2"/>
                    <path d="M20 48 Q60 68 100 48" fill="#E89B9B" stroke="#D67A7A" stroke-width="2"/>
                    <ellipse cx="60" cy="36" rx="15" ry="15" fill="#5B7C99"/>
                    <circle cx="60" cy="36" r="6" fill="#2C3E50"/>
                    <circle cx="63" cy="33" r="2" fill="#FFF"/>
                </svg>
                <div style='color:#0E7C7B; font-weight:700; font-size:0.85rem; margin-top:0.4rem;'>✓ BENAR</div>
                <div style='color:#4A615E; font-size:0.78rem; margin-top:0.2rem;'>Konjungtiva terlihat jelas, terang, fokus tajam</div>
            </div>
            """, unsafe_allow_html=True)
        with col_salah:
            st.markdown("""
            <div style='background:#FFFFFF; border:2px solid #E07A7A; border-radius:16px; padding:1rem; text-align:center;'>
                <svg viewBox="0 0 120 80" width="100%" height="90">
                    <ellipse cx="60" cy="40" rx="50" ry="28" fill="#3A3A3A" opacity="0.5"/>
                    <ellipse cx="60" cy="40" rx="50" ry="28" fill="none" stroke="#E07A7A" stroke-width="2"/>
                    <ellipse cx="60" cy="38" rx="18" ry="18" fill="#5B7C99" opacity="0.6"/>
                    <circle cx="60" cy="38" r="7" fill="#2C3E50" opacity="0.6"/>
                    <rect x="0" y="0" width="120" height="80" fill="#1C1C1C" opacity="0.25"/>
                </svg>
                <div style='color:#E07A7A; font-weight:700; font-size:0.85rem; margin-top:0.4rem;'>✗ SALAH</div>
                <div style='color:#4A615E; font-size:0.78rem; margin-top:0.2rem;'>Gelap, buram, konjungtiva tidak ditarik ke bawah</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>👤 Data Pasien</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: id_pasien = st.text_input("ID Pasien", placeholder="PSN-001")
    with c2: nama = st.text_input("Nama Samaran", placeholder="Ibu A")
    with c3: usia = st.number_input("Usia", min_value=0, max_value=120, value=25)

    if id_pasien:
        riwayat = ambil_riwayat(id_pasien)
        if riwayat:
            st.markdown("<div class='info-box'><b>📋 Riwayat pasien ini ditemukan:</b></div>", unsafe_allow_html=True)
            for tgl, hsl, conf in riwayat:
                st.markdown(f"<div class='list-item'><b>{tgl}</b> &mdash; {hsl} ({round(conf*100)}% keyakinan)</div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>📷 Unggah Foto Konjungtiva</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Unggah foto", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Foto yang diunggah", width=260)
        if st.button("🔍 Periksa Sekarang"):
            if not id_pasien:
                st.warning("Mohon isi ID Pasien terlebih dahulu.")
                return
            with st.spinner("Menganalisis dengan ensemble 3 model..."):
                pa = prediksi_ensemble(models, img)
                h = interpretasi(pa)

            st.markdown(f"""
            <div class='hasil-card {h["card"]}'>
                <p class='hasil-judul'>{h["judul"]}</p>
                <p class='hasil-pct'>Keyakinan: {h["pct"]}%</p>
                <p class='hasil-pesan'>{h["pesan"]}</p>
            </div>""", unsafe_allow_html=True)

            pa_pct = round(pa*100); normal_pct = 100 - pa_pct
            st.markdown(f"""
            <div class='prob-panel'>
                <div class='prob-title'>Rincian Probabilitas Model</div>
                <div class='prob-row'><span class='prob-label'>Indikasi Anemia</span><span class='prob-val'>{pa_pct}%</span></div>
                <div class='prob-bar-bg'><div class='prob-bar-fill anemia' style='width:{pa_pct}%'></div></div>
                <div class='prob-row' style='margin-top:0.6rem;'><span class='prob-label'>Kondisi Normal</span><span class='prob-val'>{normal_pct}%</span></div>
                <div class='prob-bar-bg'><div class='prob-bar-fill normal' style='width:{normal_pct}%'></div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div class='sec-head'>💡 Saran Tindak Lanjut</div>", unsafe_allow_html=True)
            for r in h["rekomendasi"]:
                st.markdown(f"<div class='list-item'>{r}</div>", unsafe_allow_html=True)

            label = {"anemia":"Indikasi Anemia","ragu":"Belum Pasti","normal":"Normal"}[h["status"]]
            simpan_pemeriksaan(id_pasien, nama, usia, label, h["conf_simpan"], st.session_state.petugas)
            st.success("Hasil tersimpan ke database.")
            st.markdown("<div class='disclaimer'>⚠️ <b>Penting:</b> CONJUNET adalah alat bantu skrining awal, "
                        "BUKAN diagnosis. Hasil tidak menggantikan pemeriksaan tenaga medis. Saran gizi bersifat "
                        "edukasi umum, bukan resep pengobatan.</div>", unsafe_allow_html=True)

# ── HALAMAN: EDUKASI ──────────────────────────────
def page_edukasi():
    st.markdown("""
    <div class='hero-photo hero-edu-bg'>
        <div class='hero-logo'>📚 EDUKASI</div>
        <div class='hero-title'>Cegah Anemia<br>Mulai dari Piring</div>
        <p class='hero-sub'>Pengetahuan gizi sederhana untuk membantu masyarakat
        mencegah dan memulihkan kondisi anemia.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>Apa itu Anemia?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    Anemia adalah kondisi ketika tubuh kekurangan sel darah merah sehat atau hemoglobin
    untuk membawa oksigen ke seluruh tubuh. Pada ibu hamil, anemia meningkatkan risiko
    kelahiran prematur dan bayi berat lahir rendah. Gejala umumnya meliputi lemas, pusing,
    kulit pucat, dan mudah lelah.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>🥗 Makanan Kaya Zat Besi</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    foods = [("🥬","Bayam"),("🥩","Daging Merah"),("🥚","Telur"),("🫘","Kacang")]
    for col,(emoji,nama) in zip([c1,c2,c3,c4], foods):
        with col:
            st.markdown(f"<div class='food-card'><div class='food-emoji'>{emoji}</div><div class='food-name'>{nama}</div></div>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    foods2 = [("🍓","Buah Vit C"),("🐟","Ikan"),("🍤","Hati Ayam"),("🥦","Brokoli")]
    for col,(emoji,nama) in zip([c5,c6,c7,c8], foods2):
        with col:
            st.markdown(f"<div class='food-card'><div class='food-emoji'>{emoji}</div><div class='food-name'>{nama}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>💡 Tips Penyerapan Zat Besi</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='list-item'>✅ Konsumsi vitamin C (jeruk, jambu, tomat) bersama makanan berzat besi untuk meningkatkan penyerapan.</div>
    <div class='list-item'>✅ Konsumsi tablet tambah darah (TTD) sesuai anjuran bidan, terutama bagi ibu hamil.</div>
    <div class='list-item'>⚠️ Hindari minum teh atau kopi tepat setelah makan, karena menghambat penyerapan zat besi.</div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card-teal'>📌 Informasi ini bersifat edukasi gizi umum. "
                "Untuk penanganan medis, selalu konsultasikan dengan tenaga kesehatan.</div>",
                unsafe_allow_html=True)

# ── HALAMAN: RIWAYAT ──────────────────────────────
def page_riwayat():
    st.markdown("""
    <div class='hero-photo hero-hist-bg'>
        <div class='hero-logo'>📋 RIWAYAT</div>
        <div class='hero-title'>Catatan Pemeriksaan</div>
        <p class='hero-sub'>Pantau riwayat skrining seluruh pasien yang telah diperiksa.</p>
    </div>
    """, unsafe_allow_html=True)

    cari = st.text_input("🔍 Cari berdasarkan ID Pasien", placeholder="Kosongkan untuk lihat semua")

    if cari:
        rows = ambil_riwayat(cari)
        rows = [(cari, "-", t, h, c) for (t, h, c) in rows]
    else:
        rows = semua_riwayat()

    if not rows:
        st.markdown("<div class='info-box'>Belum ada data pemeriksaan tersimpan.</div>", unsafe_allow_html=True)
        return

    st.markdown(f"<div class='sec-caption'>Menampilkan {len(rows)} catatan terakhir</div>", unsafe_allow_html=True)
    for row in rows:
        idp, nm, tgl, hsl, conf = row
        warna = {"Indikasi Anemia":"#E07A7A","Belum Pasti":"#E8A87C","Normal":"#0E7C7B"}.get(hsl, "#41B3A3")
        st.markdown(f"""
        <div class='list-item' style='border-left-color:{warna};'>
            <b>{idp}</b> {('· '+nm) if nm and nm!='-' else ''} <br>
            <span style='font-size:0.82rem;'>{tgl} &mdash; <b style='color:{warna};'>{hsl}</b> ({round(conf*100)}% keyakinan)</span>
        </div>""", unsafe_allow_html=True)

# ── HALAMAN: STATISTIK ────────────────────────────
def page_statistik():
    st.markdown("""
    <div class='hero-photo hero-stat-bg'>
        <div class='hero-logo'>🗺️ STATISTIK DAERAH</div>
        <div class='hero-title'>Peta Kerentanan<br>Anemia Indonesia</div>
        <p class='hero-sub'>Analisis klasterisasi 34 provinsi mengungkap bahwa beban anemia
        tidak merata dan butuh pendekatan berbeda tiap zona.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>Tiga Zona Kerentanan</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='zona-card zona-paradoks'>
        <div class='zona-nama'>🟠 Zona Paradoks</div>
        <div class='zona-detail'>Kondisi struktural baik (akses air 88,9%, kemiskinan rendah)
        namun BBLR tertinggi (4,88%). Menandakan adanya kesenjangan kepatuhan di tingkat individu
        yang luput dari data agregat. Butuh deteksi presisi individual.</div>
    </div>
    <div class='zona-card zona-kritis'>
        <div class='zona-nama'>🔴 Zona Kritis Struktural</div>
        <div class='zona-detail'>Beban tinggi disertai keterbatasan infrastruktur kesehatan.
        BBLR 4,84% dengan tantangan akses layanan. Butuh penguatan fasilitas sekaligus skrining.</div>
    </div>
    <div class='zona-card zona-aman'>
        <div class='zona-nama'>🟢 Zona Relatif Aman</div>
        <div class='zona-detail'>Kondisi kesehatan dan struktural relatif lebih baik.
        Tetap memerlukan pemantauan rutin untuk mencegah kemunduran.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>📊 Klasterisasi K-Means pada empat indikator (BBLR, cakupan TTD, "
                "kemiskinan, akses air minum) divalidasi dengan metode Elbow, Silhouette, dan PCA, "
                "membuktikan disparitas beban anemia antardaerah secara kuantitatif.</div>",
                unsafe_allow_html=True)

# ── ROUTER UTAMA ──────────────────────────────────
def halaman_utama():
    sig = _signatures()
    
    # Download model dari HuggingFace kalau belum ada (cloud deployment)
    if not ensure_models():
        st.error("Gagal memuat model. Coba refresh halaman.")
        return
    
    models, missing = muat_ensemble(sig)

    # Bar atas: nama petugas + keluar
    cc1, cc2 = st.columns([4,1])
    with cc2:
        if st.button("Keluar", use_container_width=True):
            st.session_state.login = False
            st.session_state.petugas = None
            st.session_state.menu = "Beranda"
            st.rerun()

    navigasi()

    if st.session_state.menu == "Skrining" and missing:
        st.error(f"File model tidak ditemukan: {', '.join(missing)}. Pastikan ketiga file .keras ada di folder.")
        return

    menu = st.session_state.menu
    if menu == "Beranda":     page_beranda()
    elif menu == "Skrining":  page_skrining(models)
    elif menu == "Edukasi":   page_edukasi()
    elif menu == "Riwayat":   page_riwayat()
    elif menu == "Statistik": page_statistik()

# ── ENTRY ─────────────────────────────────────────
if st.session_state.login:
    halaman_utama()
else:
    halaman_login()
