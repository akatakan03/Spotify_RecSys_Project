import os
import time
import polars as pl
import yt_dlp
import imageio_ffmpeg  # BİZİ KURTARAN YENİ KÜTÜPHANE

# FFMPEG'in Python içindeki gizli yerini buluyoruz
ffmpeg_yolu = imageio_ffmpeg.get_ffmpeg_exe()

input_file = 'data/processed/offline_track_metadata.parquet'
df_metadata = pl.read_parquet(input_file)

# Test için sadece ilk 50 şarkı
test_df = df_metadata.head(50)

audio_folder = 'data/audio_snippets'
os.makedirs(audio_folder, exist_ok=True)

print(f"Toplam {len(test_df)} şarkı için YouTube'dan 30 saniyelik MP3'ler indiriliyor...")
print(f"Kullanılan motor: {ffmpeg_yolu}")

for row in test_df.iter_rows(named=True):
    track_name = row['track_name']
    artist_name = row['artist_name']
    uri = row['track_uri']
    
    clean_id = uri.replace('spotify:track:', '')
    final_filename = os.path.join(audio_folder, f"{clean_id}.mp3")
    
    if os.path.exists(final_filename):
        print(f"[Zaten Var] {track_name} atlanıyor.")
        continue

    search_query = f"ytsearch1:{track_name} {artist_name} audio"
    print(f"\n[İşlemde] {track_name} - {artist_name}...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': False,
        'outtmpl': f'{audio_folder}/{clean_id}.%(ext)s',
        'ffmpeg_location': ffmpeg_yolu,  # İŞTE SİHİRLİ DOKUNUŞ BURASI!
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'postprocessor_args': {
            'FFmpegExtractAudio': ['-ss', '00:00:00', '-t', '00:00:30']
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(search_query, download=True)
        print("-> Başarıyla 30 sn MP3 olarak kaydedildi!")
        time.sleep(1) 
        
    except Exception as e:
        print(f"[Hata] {track_name} işlemi başarısız: {e}")
        time.sleep(3)

print("\nTüm işlemler tamamlandı! Klasörü kontrol edebilirsin.")