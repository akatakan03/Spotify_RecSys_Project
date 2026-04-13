import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# 1. Verilerimizi yüklüyoruz (V2 dosyasını kullanıyoruz!)
print("Veriler birleştiriliyor...")
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
df_features = pl.read_parquet('data/processed/audio_features_v2.parquet')

# 2. THE GRAND JOIN (İki tabloyu track_uri üzerinden birleştiriyoruz)
df_merged = df_features.join(df_metadata, on='track_uri', how='inner')
df = df_merged.to_pandas()

# 3. İleri Düzey Normalizasyon (StandardScaler)
# MFCC değerleri eksili (-) sayılar olabildiği için yapay zekanın kafası karışmasın diye 
# tüm değerleri Z-Skoru (ortalama 0, standart sapma 1) yöntemine göre standartlaştırıyoruz.
print("11 Boyutlu ses özellikleri yapay zeka için ölçeklendiriliyor...")
scaler = StandardScaler()
feature_cols = [
    'bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 
    'spectral_rolloff', 'harmonic_ratio', 
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5'
]
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 4. Gelişmiş Öneri Motoru Fonksiyonu
def sarki_oner(hedef_sarki_adi, onerilecek_sayi=3):
    # Şarkıyı listede bulalım (Büyük/küçük harf duyarsız arama)
    hedef_sarki = df[df['track_name'].str.contains(hedef_sarki_adi, case=False, na=False)]
    
    if hedef_sarki.empty:
        return f"Hata: '{hedef_sarki_adi}' isimli şarkı bizim test listemizde bulunamadı."
    
    hedef_index = hedef_sarki.index[0]
    tam_isim = hedef_sarki['track_name'].values[0]
    sanatci = hedef_sarki['artist_name'].values[0]
    
    print(f"\n🎧 Seçilen Şarkı: {tam_isim} - {sanatci}")
    print("-" * 50)
    
    # 5. Benzerlik Hesaplama (Cosine Similarity)
    # 11 boyutlu uzayda şarkıların birbirine olan açısını (benzerliğini) ölçüyoruz
    tum_ozellikler = df[feature_cols].values
    benzerlik_matrisi = cosine_similarity(tum_ozellikler)
    
    # Bizim şarkımızın diğerleriyle olan benzerlik skorlarını alıyoruz
    skorlar = list(enumerate(benzerlik_matrisi[hedef_index]))
    
    # Skorları büyükten küçüğe sıralıyoruz
    sirali_skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
    
    # Kendisini listeden çıkarıp en yakın şarkıları alıyoruz
    en_yakinlar = sirali_skorlar[1 : onerilecek_sayi + 1]
    
    # Sonuçları ekrana basıyoruz
    print("Müzikal Olarak (11 Özelliğe Göre) En Çok Önerilenler:")
    for i, (idx, skor) in enumerate(en_yakinlar):
        onerilen_isim = df.iloc[idx]['track_name']
        onerilen_sanatci = df.iloc[idx]['artist_name']
        
        # Cosine Similarity -1 ile 1 arasıdır. Yüzdeye çevirmek için ufak bir matematik:
        benzerlik_yuzdesi = ((skor + 1) / 2) * 100 
        
        print(f"{i+1}. {onerilen_isim} - {onerilen_sanatci} (Benzerlik: %{benzerlik_yuzdesi:.1f})")

# TEST AŞAMASI
# df.iloc[0] diyerek listedeki ilk şarkıyı otomatik seçiyoruz. 
# İstersen buraya doğrudan bir şarkı adı yazabilirsin: sarki_oner("Master of Puppets")
test_sarkisi = df.iloc[0]['track_name'] 
sarki_oner(test_sarkisi)