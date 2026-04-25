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

for index in range(len(df)):
    # Bir şarkının diğer tüm şarkılarla olan benzerlik skorlarını al
    skorlar = list(enumerate(benzerlik_matrisi[index]))
    
    # Büyükten küçüğe sırala
    sirali_skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
    
    # Kendisini (1.0) atla ve en yakın 3 komşuyu al
    en_yakinlar = sirali_skorlar[1 : k_komsuluk + 1] 
    
    # Bu 3 komşunun benzerlik skorlarını % formatına çevir ve ortalamasını al
    sarki_ortalama_skoru = np.mean([((skor + 1) / 2) * 100 for idx, skor in en_yakinlar])
    
    # Sistemin genel toplamına ekle
    toplam_benzerlik += sarki_ortalama_skoru

# 5. Sistemin Genel Başarı Notunu (Global Confidence) Hesaplama
genel_ortalama = toplam_benzerlik / len(df)

print("-" * 50)
print(f"🎯 SİSTEMİN GENEL BAŞARI (CONFIDENCE) SKORU: %{genel_ortalama:.2f}")
print("-" * 50)

# Sonucu Yorumlama (Thresholds)
if genel_ortalama > 85:
    print("Sonuç: MÜKEMMEL! Model şarkılar arasında çok güçlü müzikal bağlar kurabiliyor.")
elif genel_ortalama > 75:
    print("Sonuç: BAŞARILI. Öneriler tutarlı ancak veri seti büyüdükçe daha da iyileşecektir.")
else:
    print("Sonuç: DÜŞÜK. Veri havuzu (Cold Start) çok küçük olduğu için model zorlanıyor.")