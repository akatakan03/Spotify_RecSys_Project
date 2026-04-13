import os
import json
import polars as pl

# 1. Hedef 100 bin şarkımızı RAM'e alıyoruz
subset_file = 'data/subset_100k_tracks.parquet'
df_subset = pl.read_parquet(subset_file)

# Hızlı arama (O(1) complexity) yapabilmek için listeyi Set'e çeviriyoruz
target_uris = set(df_subset['track_uri'].to_list()) 

# 2. Çıkarılacak verileri tutacağımız Dictionary (Sözlük)
extracted_metadata = {}

# JSON dosyalarının bulunduğu klasör
json_folder = 'data/raw/' 
dosyalar = [f for f in os.listdir(json_folder) if f.endswith('.json')]

print("Offline veri çıkarımı başlıyor. İnternet yok, API yok, banlanma yok!")

for i, dosya in enumerate(dosyalar):
    with open(os.path.join(json_folder, dosya), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for playlist in data['playlists']:
            for track in playlist['tracks']:
                uri = track['track_uri']
                
                # Eğer bu şarkı bizim 100 binlik listedeyse
                if uri in target_uris:
                    if uri not in extracted_metadata:
                        # Şarkıyı ilk kez görüyorsak kaydet
                        extracted_metadata[uri] = {
                            'track_uri': uri,
                            'track_name': track['track_name'],
                            'artist_name': track['artist_name'],
                            'album_name': track['album_name'],
                            'duration_ms': track['duration_ms'],
                            'playlist_count': 1 # Popülarite (Occurrence) değerimiz
                        }
                    else:
                        # Şarkıyı daha önce kaydettiysek sadece popülaritesini artır
                        extracted_metadata[uri]['playlist_count'] += 1

    if (i + 1) % 100 == 0:
        print(f"Tarama İlerlemesi: {i+1}/{len(dosyalar)} dosya bitti.")

# 3. Sonuçları tabloya (DataFrame) çevirip diske kaydediyoruz
if extracted_metadata:
    df_final = pl.DataFrame(list(extracted_metadata.values()))
    output_path = 'data/processed/offline_track_metadata.parquet'
    
    # Klasör yoksa oluştur
    os.makedirs('data/processed', exist_ok=True)
    
    df_final.write_parquet(output_path)
    print(f"\nMükemmel! {len(df_final)} şarkının metadata bilgisi sıfır hatayla çıkarıldı.")
    print(f"Dosya şuraya kaydedildi: {output_path}")
else:
    print("Hiçbir eşleşme bulunamadı. JSON klasör yolunu kontrol edin.")