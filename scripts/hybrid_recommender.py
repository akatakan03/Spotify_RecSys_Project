import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

print("Hybrid Recommendation Engine Başlatılıyor...\n")

# 1. Verileri yüklüyoruz ve birleştiriyoruz (The Grand Join)
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
df_features = pl.read_parquet('data/processed/audio_features_v2.parquet')
df_merged = df_features.join(df_metadata, on='track_uri', how='inner')
df = df_merged.to_pandas()

# 2. Audio Features için Standardization
audio_cols = [
    'bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 
    'spectral_rolloff', 'harmonic_ratio', 
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5'
]
audio_scaler = StandardScaler()
df[audio_cols] = audio_scaler.fit_transform(df[audio_cols])

# 3. Popülarite için Normalization (0 ile 1 arasına çekme)
pop_scaler = MinMaxScaler()
df['normalized_popularity'] = pop_scaler.fit_transform(df[['playlist_count']])

# 4. Audio Benzerlik Matrisi (Cosine Similarity)
audio_features_matrix = df[audio_cols].values
audio_similarity_matrix = cosine_similarity(audio_features_matrix)

# 5. Hybrid Recommendation Fonksiyonu
def hybrid_recommendation(target_song_name, top_n=3, audio_weight=0.75, pop_weight=0.25):
    hedef_sarki = df[df['track_name'].str.contains(target_song_name, case=False, na=False)]
    
    if hedef_sarki.empty:
        return f"Hata: '{target_song_name}' bulunamadı."
        
    hedef_index = hedef_sarki.index[0]
    
    print(f"🎧 Seçilen Şarkı: {hedef_sarki['track_name'].values[0]} - {hedef_sarki['artist_name'].values[0]}")
    print(f"⚙️ Kullanılan Algoritma Ağırlıkları: %{int(audio_weight*100)} Ses Analizi, %{int(pop_weight*100)} İnsan Popülaritesi")
    print("-" * 70)
    
    scores = []
    
    for idx in range(len(df)):
        if idx == hedef_index:
            continue # Şarkının kendisini önermiyoruz
            
        # Cosine Similarity -1 ile 1 arasındadır, bunu 0-1 aralığına çekiyoruz
        audio_sim = (audio_similarity_matrix[hedef_index][idx] + 1) / 2
        
        # Daha önce 0-1 arasına sıkıştırdığımız popülarite skoru
        popularity = df.iloc[idx]['normalized_popularity']
        
        # THE HYBRID EQUATION (Hibrit Denklem)
        hybrid_score = (audio_sim * audio_weight) + (popularity * pop_weight)
        
        scores.append((idx, hybrid_score, audio_sim, popularity))
        
    # Skorları büyükten küçüğe sıralıyoruz (Sorting)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # En iyi sonuçları ekrana basıyoruz
    print("🏆 En İyi Hibrit Öneriler:")
    for i, (idx, h_score, a_score, p_score) in enumerate(scores[:top_n]):
        isim = df.iloc[idx]['track_name']
        sanatci = df.iloc[idx]['artist_name']
        print(f"{i+1}. {isim} - {sanatci}")
        print(f"   -> Toplam Skor: %{h_score*100:.1f} (Ses Benzerliği: %{a_score*100:.1f} | Popülarite Çarpanı: %{p_score*100:.1f})\n")

# TEST AŞAMASI
test_sarkisi = df.iloc[0]['track_name']
hybrid_recommendation(test_sarkisi)