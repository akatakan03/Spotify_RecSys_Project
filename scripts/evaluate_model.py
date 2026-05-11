import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Terminalimizi temiz tutmak için uyarıları gizliyoruz
warnings.filterwarnings('ignore')

print("Model Değerlendirme (Evaluation) Başlıyor...\n")

# 1. Verileri Yükleme ve Birleştirme
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
df_features = pl.read_parquet('data/processed/audio_features_v2.parquet')
df_merged = df_features.join(df_metadata, on='track_uri', how='inner')
df = df_merged.to_pandas()

# 2. Ölçeklendirme (Standardization)
scaler = StandardScaler()
feature_cols = [
    'bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 
    'spectral_rolloff', 'harmonic_ratio', 
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5'
]
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 3. Benzerlik Matrisini (Similarity Matrix) Oluşturma
tum_ozellikler = df[feature_cols].values
benzerlik_matrisi = cosine_similarity(tum_ozellikler)

# 4. Sınav (Evaluation) Aşaması
toplam_benzerlik = 0
k_komsuluk = 3 # Her şarkı için en iyi 3 öneriyi (Top-K) test edeceğiz

print(f"Sistemdeki {len(df)} şarkının her biri için Top-{k_komsuluk} öneri test ediliyor...")

# YENİ EKLENEN: Çeşitlilik (Diversity) Takibi İçin
onerilen_essiz_sanatcilar = set()
sistemdeki_toplam_sanatcilar = df['artist_name'].nunique()

for index in range(len(df)):
    # Bir şarkının diğer tüm şarkılarla olan benzerlik skorlarını al
    skorlar = list(enumerate(benzerlik_matrisi[index]))
    sirali_skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
    en_yakinlar = sirali_skorlar[1 : k_komsuluk + 1] 
    
    # Benzerlik skorlarını hesapla
    sarki_ortalama_skoru = np.mean([((skor + 1) / 2) * 100 for idx, skor in en_yakinlar])
    toplam_benzerlik += sarki_ortalama_skoru
    
    # YENİ EKLENEN: Önerilen şarkıların sanatçılarını havuza ekle
    for idx, skor in en_yakinlar:
        onerilen_sanatci = df.iloc[idx]['artist_name']
        onerilen_essiz_sanatcilar.add(onerilen_sanatci)

# 5. Sistemin Genel Başarı Notlarını (Metrics) Hesaplama
genel_ortalama = toplam_benzerlik / len(df)

# YENİ EKLENEN: Catalog Coverage (Katalog Kapsayıcılığı) Formülü
catalog_coverage = (len(onerilen_essiz_sanatcilar) / sistemdeki_toplam_sanatcilar) * 100

print("-" * 70)
print(f"🎯 MÜZİKAL UYUM (CONFIDENCE) SKORU: %{genel_ortalama:.2f}")
print(f"🌍 CATALOG COVERAGE (Çeşitlilik): %{catalog_coverage:.2f} (Toplam {sistemdeki_toplam_sanatcilar} sanatçıdan {len(onerilen_essiz_sanatcilar)} tanesi önerildi)")
print("-" * 70)

# Sonucu Yorumlama
if catalog_coverage > 20: 
    print("Sonuç: HARİKA! Sistem sadece hit parçaları değil, kütüphanedeki gizli kalmış (Long-tail) sanatçıları da keşfedip öneriyor. Popularity Bias başarıyla kırıldı.")
else:
    print("Sonuç: DİKKAT. Çeşitlilik düşük. Sistem hep aynı popüler sanatçıların etrafında dönüyor olabilir.")