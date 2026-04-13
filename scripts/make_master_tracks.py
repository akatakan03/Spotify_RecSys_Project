import os
import json
import polars as pl

# 1. Klasör yolunu belirliyoruz
data_path = 'data/raw/' # JSON dosyalarının olduğu klasör
unique_tracks = set()

# 2. Klasördeki tüm dosyaları tek tek geziyoruz
files = [f for f in os.listdir(data_path) if f.endswith('.json')]

for file_name in files:
    with open(os.path.join(data_path, file_name), 'r') as f:
        data = json.load(f)
        
        # 3. Playlists ve Tracks döngüsü
        for playlist in data['playlists']:
            for track in playlist['tracks']:
                unique_tracks.add(track['track_uri'])

# 4. Sonucu bir listeye çevirip Polars DataFrame oluşturuyoruz
track_list = list(unique_tracks)
df_master = pl.DataFrame({"track_uri": track_list})

# 5. Veriyi hızlı okuma için Parquet formatında kaydediyoruz
df_master.write_parquet('data/master_tracks.parquet')

print(f"İşlem tamam! Toplam {len(unique_tracks)} eşsiz şarkı bulundu.")