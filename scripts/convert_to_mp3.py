import os
import subprocess

audio_folder = 'data/audio_snippets'

print("Dönüştürme işlemi başlıyor...")

for filename in os.listdir(audio_folder):
    if filename.endswith(".webm") or filename.endswith(".m4a"):
        input_path = os.path.join(audio_folder, filename)
        
        output_path = os.path.join(audio_folder, os.path.splitext(filename)[0] + ".mp3")
        
        command = [
            'ffmpeg',
            '-y', 
            '-i', input_path,
            '-ss', '00:00:00',
            '-t', '00:00:30',
            '-q:a', '0', 
            '-map', 'a', 
            output_path
        ]
        
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Başarılı] {filename} -> MP3'e çevrildi (30 sn)")
            
            os.remove(input_path)
            
        except Exception as e:
            print(f"[Hata] {filename} dönüştürülemedi: {e}")

print("Tüm dönüştürme işlemleri tamamlandı!")