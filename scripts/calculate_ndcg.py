import os
import json
import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import math
import warnings

warnings.filterwarnings('ignore')

print("Geniş Çaplı Akademik Sıralama Testi (NDCG & R-Precision) Başlıyor...\n")

# 1. Load Data (Verileri Yükleme)
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
df_features = pl.read_parquet('data/processed/audio_features_v2.parquet')
df_merged = df_features.join(df_metadata, on='track_uri', how='inner').to_pandas()

# 2. Standardization (Ölçeklendirme)
scaler = StandardScaler()
feature_cols = ['bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff', 'harmonic_ratio', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5']
df_merged[feature_cols] = scaler.fit_transform(df_merged[feature_cols])

havuz_uris = set(df_merged['track_uri'].tolist())
tum_ozellikler = df_merged[feature_cols].values

# 3. Find All Playlists (Uygun Olan TÜM Çalma Listelerini Bulma)
json_folder = 'data/raw/'
dosyalar = [f for f in os.listdir(json_folder) if f.endswith('.json')]

uygun_playlistler = []

print("Uygun çalma listeleri aranıyor...")
for dosya in dosyalar:
    with open(os.path.join(json_folder, dosya), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for playlist in data['playlists']:
            playlist_uris = [t['track_uri'] for t in playlist['tracks']]
            ortak_sarkilar = [uri for uri in playlist_uris if uri in havuz_uris]
            
            if len(ortak_sarkilar) >= 10:
                uygun_playlistler.append({
                    'name': playlist['name'],
                    'tracks': ortak_sarkilar[:10]
                })
                
    # İşlemin saatlerce sürmemesi için ilk 50 uygun listeyi bulduğumuzda taramayı durduruyoruz
    if len(uygun_playlistler) >= 50:
        break

if not uygun_playlistler:
    print("Hata: Uygun çalma listesi bulunamadı.")
    exit()

print(f"📌 Toplam {len(uygun_playlistler)} adet uygun çalma listesi bulundu. Sınav başlıyor...\n")

toplam_ndcg = 0
toplam_r_precision = 0

# 4. Evaluate Each Playlist (Her Bir Listeyi Tek Tek Sınava Sokma)
for idx, pl_data in enumerate(uygun_playlistler):
    kesisen_sarkilar = pl_data['tracks']
    seed_tracks = kesisen_sarkilar[:8]
    target_tracks = kesisen_sarkilar[8:]
    
    seed_df = df_merged[df_merged['track_uri'].isin(seed_tracks)]
    user_profile = seed_df[feature_cols].mean().values.reshape(1, -1)
    
    benzerlik_skorlari = cosine_similarity(user_profile, tum_ozellikler)[0]
    
    skorlar = list(enumerate(benzerlik_skorlari))
    sirali_skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
    
    oneriler = []
    for s_idx, skor in sirali_skorlar:
        uri = df_merged.iloc[s_idx]['track_uri']
        if uri not in seed_tracks:
            oneriler.append(uri)
            
    top_k_oneriler = oneriler[:50]
    
    # R-Precision
    dogru_tahminler = set(top_k_oneriler).intersection(set(target_tracks))
    r_prec = len(dogru_tahminler) / len(target_tracks)
    toplam_r_precision += r_prec
    
    # NDCG
    dcg = 0
    idcg = 0
    
    for i, uri in enumerate(top_k_oneriler):
        if uri in target_tracks:
            dcg += 1 / math.log2(i + 1 + 1)
            
    for i in range(len(target_tracks)):
        idcg += 1 / math.log2(i + 1 + 1)
        
    ndcg = dcg / idcg if idcg > 0 else 0
    toplam_ndcg += ndcg

# 5. Average Scores (Genel Ortalamayı Alma)
ortalama_r_precision = toplam_r_precision / len(uygun_playlistler)
ortalama_ndcg = toplam_ndcg / len(uygun_playlistler)

print("-" * 50)
print(f"📊 ORTALAMA R-Precision Skoru: {ortalama_r_precision:.4f}")
print(f"🥇 ORTALAMA NDCG Skoru: {ortalama_ndcg:.4f}")
print("-" * 50)