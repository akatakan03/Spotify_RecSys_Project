import streamlit as st
import sys
import os

# --- ALTYAPI BAĞLANTILARI ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from recommender_pipeline import SpotifyRecommenderPipeline
from spotify_api import SpotifyAPIHelper

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Popular | Hear the unheard.", page_icon="🎧", layout="centered")

# --- 2. ÖZEL CSS (FLUTTER 'POPULAR' TEMASI) ---
st.markdown("""
<style>
    /* Streamlit'in varsayılan üst menüsünü ve alt bilgisini gizle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Ana Arka Plan */
    .stApp {
        background-color: #0A0A0A;
    }

    /* Popular Marka Başlığı */
    .brand-text {
        font-size: 4rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #26C6DA, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -15px;
        letter-spacing: -2px;
        text-align: center;
    }
    
    .brand-slogan {
        color: #26C6DA;
        font-style: italic;
        text-align: center;
        margin-bottom: 50px;
        font-size: 1.2rem;
    }

    /* Gradyan Buton Tasarımı */
    div.stButton > button {
        background: linear-gradient(45deg, #26C6DA, #8BC34A);
        color: #000000 !important;
        border-radius: 16px;
        border: none;
        padding: 12px 24px;
        font-weight: 900;
        font-size: 16px;
        width: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 15px rgba(38, 198, 218, 0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 195, 74, 0.4);
    }

    /* Şarkı Kartları */
    .song-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 15px;
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
        transition: background-color 0.3s;
    }
    .song-card:hover {
        background-color: rgba(255, 255, 255, 0.06);
    }
    
    .song-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-artist {
        color: #8BC34A;
        font-size: 0.9rem;
        margin: 0;
    }
    .song-score {
        background-color: rgba(0,0,0,0.5);
        border: 1px solid #26C6DA;
        color: #26C6DA;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* Girdi Kutusu (Input Field) Özel Tasarımı */
    input[type="text"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(38, 198, 218, 0.5) !important;
        border-radius: 16px !important;
        color: white !important;
        padding: 15px !important;
        font-size: 1.1rem !important;
        text-align: center;
    }
    input[type="text"]:focus {
        border: 1px solid #8BC34A !important;
        box-shadow: 0 0 10px rgba(139, 195, 74, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODELLERİ YÜKLEME ---
CLIENT_ID = "7e58c74eb38f481c918fcda316eed479"
CLIENT_SECRET = "c4c97cb892de4d48bd738e151dd88f67"

@st.cache_resource
def load_models():
    pipeline = SpotifyRecommenderPipeline()
    spotify = SpotifyAPIHelper(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return pipeline, spotify

# Marka Başlığı
st.markdown('<div class="brand-text">Popular</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-slogan">Hear the unheard.</div>', unsafe_allow_html=True)

with st.spinner("Müzik motoru ısınıyor..."):
    pipeline, spotify_api = load_models()

def sarkilari_goster(sonuclar):
    st.markdown("<br>", unsafe_allow_html=True)
    for sarki in sonuclar:
        live_data = spotify_api.get_track_details(sarki['uri'])
        clean_id = sarki['uri'].split(':')[-1] 
        yerel_ses_dosyasi = f"data/audio_snippets/{clean_id}.mp3" 
        
        kapak_url = live_data['cover_url'] if live_data and live_data['cover_url'] else "https://www.scdn.co/mirror/static/images/fallback/default-track-artwork.png"
        skor_yuzde = int(sarki['score'] * 100)
        
        html_card = f"""
        <div class="song-card">
            <img src="{kapak_url}" width="65" height="65" style="border-radius: 12px; object-fit: cover;">
            <div style="flex-grow: 1; overflow: hidden;">
                <p class="song-title">{sarki['name']}</p>
                <p class="song-artist">{sarki['artist']}</p>
                <div class="song-score">% {skor_yuzde} Eşleşme</div>
            </div>
        </div>
        """
        
        colA, colB = st.columns([3, 1])
        with colA:
            st.markdown(html_card, unsafe_allow_html=True)
        with colB:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if os.path.exists(yerel_ses_dosyasi):
                st.audio(yerel_ses_dosyasi, format="audio/mp3")
            else:
                if live_data and live_data['spotify_url']:
                    st.markdown(f"<a href='{live_data['spotify_url']}' target='_blank' style='color:#1DB954; font-weight:bold; text-decoration:none; display:block; text-align:center; padding-top:10px;'>Spotify'da Aç ↗</a>", unsafe_allow_html=True)

# --- 4. TEK EKRANLI AKILLI ARAMA (UNIFIED SEARCH) ---
st.markdown("<h4 style='color: white; text-align:center;'>Şu an ne dinlemek istiyorsun?</h4>", unsafe_allow_html=True)
st.markdown("<p style='color: #b3b3b3; text-align:center; font-size: 14px;'>Bir şarkı adı veya ruh halini (Örn: 'Starboy' veya 'gece yolculuğu') yaz, gerisini yapay zekaya bırak.</p>", unsafe_allow_html=True)

kullanici_sorgusu = st.text_input("Arama Kutusu", placeholder="Şarkı adı veya ruh halin...", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Benim İçin Keşfet"):
    if kullanici_sorgusu:
        # 1. AKILLI YÖNLENDİRME (Smart Routing): Şarkı mı, Ruh Hali mi?
        arama_metni = kullanici_sorgusu.strip().lower()
        
        # DİKKAT: Artık metadata'ya değil, doğrudan SADECE sesi analiz edilmiş şarkılara (df_merged) bakıyoruz!
        matched_songs = pipeline.df_merged[pipeline.df_merged['track_name'].str.lower() == arama_metni]
        
        if not matched_songs.empty:
            # EĞER ŞARKIYSA: Hibrit Motoru Çalıştır (Bilgi mesajları kaldırıldı)
            secilen_sarki = matched_songs.iloc[0]
            secilen_uri = secilen_sarki['track_uri']
            
            sonuclar = pipeline.recommend_hybrid([secilen_uri], w_audio=0.2, w_co=0.8, top_n=5)
            sarkilari_goster(sonuclar)
            
        else:
            # EĞER ŞARKI DEĞİLSE: NLP Senaryo Motorunu Çalıştır (Bilgi mesajları kaldırıldı)
            sonuclar = pipeline.recommend_scenario(kullanici_sorgusu, top_n=5)
            if sonuclar:
                sarkilari_goster(sonuclar)
            else:
                st.warning("Bu hisse tam uyan bir parça bulamadık, farklı kelimelerle tarif edebilir misin?")
    else:
        st.warning("Lütfen arama kutusuna bir şeyler yazın.")