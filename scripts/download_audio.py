import os
import time
import random
import polars as pl
import yt_dlp
import imageio_ffmpeg

ffmpeg_yolu = imageio_ffmpeg.get_ffmpeg_exe()

input_file = 'data/processed/offline_track_metadata.parquet'
df_metadata = pl.read_parquet(input_file)

# GECE UÇUŞU: İlk 5.000 şarkıyı alıyoruz
target_df = df_metadata.head(5000)

audio_folder = 'data/audio_snippets'
os.makedirs(audio_folder, exist_ok=True)

print(f"BÜYÜK İNDİRME BAŞLIYOR: Toplam {len(target_df)} şarkı...")
print(f"Kullanılan motor: {ffmpeg_yolu}")

for index, row in enumerate(target_df.iter_rows(named=True)):
    track_name = row['track_name']
    artist_name = row['artist_name']
    uri = row['track_uri']
    
    clean_id = uri.replace('spotify:track:', '')
    final_filename = os.path.join(audio_folder, f"{clean_id}.mp3")
    
    if os.path.exists(final_filename):
        print(f"[{index+1}/5000] [Zaten Var] {track_name} atlanıyor.")
        continue

    search_query = f"ytsearch1:{track_name} {artist_name} audio"
    print(f"\n[{index+1}/5000] [İşlemde] {track_name} - {artist_name}...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True, # Terminal çok kalabalık olmasın diye detayları gizliyoruz
        'outtmpl': f'{audio_folder}/{clean_id}.%(ext)s',
        'ffmpeg_location': ffmpeg_yolu,
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
        print("-> Başarıyla kaydedildi!")
        
        # RANDOMIZATION: 1.5 ile 3.5 saniye arası rastgele bekleme
        bekleme_suresi = random.uniform(1.5, 3.5)
        time.sleep(bekleme_suresi)
        
    except Exception as e:
        hata_mesaji = str(e)
        
        # RATE LIMIT KONTROLÜ (HTTP 429)
        if '429' in hata_mesaji or 'Too Many Requests' in hata_mesaji:
            print(f"\n[DİKKAT! RATE LIMIT] YouTube şüphelendi. 15 dakika uyku moduna geçiliyor...")
            time.sleep(900) # 900 saniye = 15 dakika
            print("[UYANIŞ] İndirme işlemine gizlice devam ediliyor...\n")
        else:
            print(f"-> [Hata] {track_name} başarısız: {hata_mesaji}")
            time.sleep(5)

print("\nMuazzam! 5.000 şarkılık veri setinin ses dosyaları başarıyla tamamlandı.")