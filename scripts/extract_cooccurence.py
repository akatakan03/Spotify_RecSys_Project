import os
import json
import polars as pl
import pickle
from collections import defaultdict

print("Co-occurrence (Birlikte Bulunma) Analizi Başlıyor...\n")

# 1. 5000 şarkılık havuzumuzu belleğe alıyoruz
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
havuz_uris = set(df_metadata['track_uri'].to_list())

# 2. İç içe geçmiş bir Dictionary (Sözlük) oluşturuyoruz
co_occurrence = defaultdict(lambda: defaultdict(int))

json_folder = 'data/raw/'
dosyalar = [f for f in os.listdir(json_folder) if f.endswith('.json')]

print("JSON dosyaları taranıyor ve insan davranışları haritalandırılıyor...")

for i, dosya in enumerate(dosyalar):
    with open(os.path.join(json_folder, dosya), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for playlist in data['playlists']:
            # Sadece bizim 5000'lik havuzumuzda olan şarkıları filtrele
            playlist_uris = [t['track_uri'] for t in playlist['tracks'] if t['track_uri'] in havuz_uris]
            
            # Aynı listedeki şarkıların birbirleriyle olan ilişkisini (skorunu) 1 artır
            for x in range(len(playlist_uris)):
                for y in range(x + 1, len(playlist_uris)):
                    uri1 = playlist_uris[x]
                    uri2 = playlist_uris[y]
                    
                    co_occurrence[uri1][uri2] += 1
                    co_occurrence[uri2][uri1] += 1

    if (i + 1) % 50 == 0:
        print(f"İlerleme: {i+1} dosya tarandı.")

# 3. Öğrenilen insan davranışını diske kaydet
output_file = 'data/processed/co_occurrence.pkl'
with open(output_file, 'wb') as f:
    # DefaultDict yapısını normal Dict yapısına çevirerek kaydediyoruz
    pickle.dump({k: dict(v) for k, v in co_occurrence.items()}, f)

print(f"\nMuazzam! İnsan davranışları başarıyla analiz edildi.")
print(f"İlişki ağı şuraya kaydedildi: {output_file}")