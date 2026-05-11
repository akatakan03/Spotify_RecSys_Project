import streamlit as st
import sys
import os

# Altyapı dosyalarını bağlıyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from recommender_pipeline import SpotifyRecommenderPipeline
from spotify_api import SpotifyAPIHelper

# --- 1. SAYFA VE CSS MAKYAJI (UI/UX DESIGN) ---
st.set_page_config(page_title="Thesis: Spotify RecSys", page_icon="🎵", layout="wide")

# Özel CSS: Arka planı koyulaştırıp, Spotify yeşili butonlar ekliyoruz
st.markdown("""
<style>
    /* Ana arka plan */
    .stApp {
        background-color: #121212;
        color: white;
    }
    /* Butonları Spotify Yeşili yap */
    div.stButton > button {
        background-color: #1DB954;
        color: white;
        border-radius: 500px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1ed760;
        color: white;
    }
    /* Albüm kartları için tasarım */
    .track-card {
        background-color: #181818;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .track-title {
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin-bottom: 5px;
    }
    .track-artist {
        font-size: 14px;
        color: #b3b3b3;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. GÜVENLİK VE MODELLERİ YÜKLEME ---
# Buraya kendi API bilgilerini gir!
CLIENT_ID = "7e58c74eb38f481c918fcda316eed479"
CLIENT_SECRET = "c4c97cb892de4d48bd738e151dd88f67"

@st.cache_resource
def load_models():
    pipeline = SpotifyRecommenderPipeline()
    spotify = SpotifyAPIHelper(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return pipeline, spotify

with st.spinner("🤖 Yapay Zeka ve 🌐 Spotify API Yükleniyor..."):
    pipeline, spotify_api = load_models()

# --- 3. GÖRSEL ARAYÜZ (FRONTEND) ---
st.title("🎧 Akıllı Müzik Keşfi (Spotify AI)")
st.markdown("Tez Projesi: Senaryo Farkındalıklı ve Çeşitlilik Odaklı Hibrit Öneri Sistemi")

tab1, tab2 = st.tabs(["🎵 Şarkı ile Keşfet (Hibrit)", "✍️ Senaryo ile Keşfet (NLP)"])

# Şarkıları ekrana şık bir şekilde basan yardımcı fonksiyon (Render Function)
import os # Eğer dosyanın en üstünde yoksa bunu eklemeyi unutma

def sarkilari_goster(sonuclar):
    for sarki in sonuclar:
        # 1. Spotify API'den kapak ve link bilgilerini alıyoruz
        live_data = spotify_api.get_track_details(sarki['uri'])
        
        # 2. Windows uyumluluğu için URI temizleme (Clean ID)
        # 'spotify:track:2gam...' olan URI'yi '2gam...' haline getiriyoruz
        clean_id = sarki['uri'].split(':')[-1] 
        
        # 3. Güncel klasör yolunu tanımlıyoruz
        yerel_ses_dosyasi = f"data/audio_snippets/{clean_id}.mp3" 
        
        col1, col2, col3 = st.columns([1, 3, 2])
        
        with col1:
            if live_data and live_data['cover_url']:
                st.image(live_data['cover_url'], width=100)
            else:
                # Placeholder görsel (Eğer API resmi bulamazsa)
                st.image("https://www.scdn.co/mirror/static/images/fallback/default-track-artwork.png", width=100)
                
        with col2:
            st.markdown(f"<div class='track-title'>{sarki['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='track-artist'>{sarki['artist']}</div>", unsafe_allow_html=True)
            st.caption(f"Confidence Score: %{sarki['score']*100:.1f}")
            if live_data and live_data['spotify_url']:
                st.markdown(f"[Spotify'da Dinle]({live_data['spotify_url']})")
                
        with col3:
            # 4. Dosya varlık kontrolü ve yerel oynatıcı
            if os.path.exists(yerel_ses_dosyasi):
                st.audio(yerel_ses_dosyasi, format="audio/mp3")
            else:
                st.warning("Ses dosyası bulunamadı.")
                st.caption(f"Aranan dosya: {clean_id}.mp3")
                st.caption(f"Konum: {yerel_ses_dosyasi}")
        st.markdown("---")

# HİBRİT MOTOR
with tab1:
    ornek_sarkilar = pipeline.df_metadata['track_name'].head(100).tolist()
    secilen_sarki_adi = st.selectbox("Referans bir şarkı seçin:", ornek_sarkilar)
    
    colA, colB = st.columns(2)
    with colA: w_audio = st.slider("Ses Özellikleri (Content) Ağırlığı", 0.0, 1.0, 0.4, 0.1)
    with colB: w_co = st.slider("İnsan Davranışı (Collaborative) Ağırlığı", 0.0, 1.0, 0.6, 0.1)

    if st.button("Benzerlerini Keşfet"):
        secilen_uri = pipeline.df_metadata[pipeline.df_metadata['track_name'] == secilen_sarki_adi]['track_uri'].values[0]
        with st.spinner("AI matrisleri tarıyor..."):
            sonuclar = pipeline.recommend_hybrid([secilen_uri], w_audio=w_audio, w_co=w_co, top_n=5)
            sarkilari_goster(sonuclar)

# SENARYO MOTORU
with tab2:
    kullanici_metni = st.text_input("Nasıl bir ruh halindesiniz? (Örn: sad piano, energetic workout)")
    
    if st.button("Senaryoya Uygun Şarkı Bul"):
        if kullanici_metni:
            with st.spinner("Metin anlamsal olarak çözümleniyor (NLP)..."):
                sonuclar = pipeline.recommend_scenario(kullanici_metni, top_n=5)
                if sonuclar:
                    sarkilari_goster(sonuclar)
                else:
                    st.error("Eşleşme bulunamadı.")