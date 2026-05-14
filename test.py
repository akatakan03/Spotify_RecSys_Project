import sys
import os

# 1. Altyapı Bağlantısı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from recommender_pipeline import SpotifyRecommenderPipeline

print("--- YEREL TERMİNAL TESTİ BAŞLIYOR ---")

# 2. Modeli Başlatma (Initialize)
pipeline = SpotifyRecommenderPipeline()

# 3. Dinamik Şarkı Seçimi (Dynamic Selection)
secilen_sarki = pipeline.df_merged.iloc[0]
sarki_adi = secilen_sarki['track_name']
sanatci_adi = secilen_sarki['artist_name']
secilen_uri = secilen_sarki['track_uri']

print(f"\n✅ Otomatik Seçilen Şarkı: {sarki_adi} - {sanatci_adi}")

# 4. Öneri Motorunu Çalıştırma (Run Recommendation)
print("🎧 Hibrit Motor Çalışıyor (Audio: 0.2, Co-occurrence: 0.8)...")
sonuclar = pipeline.recommend_hybrid([secilen_uri], w_audio=0.2, w_co=0.8, top_n=5)

# 5. Sonuçları Yazdırma (Output)
print("\n--- ÖNERİLEN ŞARKILAR (OUTPUT) ---")
for i, sarki in enumerate(sonuclar):
    skor_yuzdesi = sarki['score'] * 100
    print(f"{i+1}. {sarki['name']} - {sarki['artist']} (Skor: %{skor_yuzdesi:.1f})")