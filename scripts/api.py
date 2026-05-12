import sys
import os

# Sistemin yol bulma (*Pathfinding*) işlemi. Bu satırların en üstte olması zorunludur!
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recommender_pipeline import SpotifyRecommenderPipeline

app = Flask(__name__)
CORS(app)

print("🚀 Model belleğe yükleniyor, lütfen bekleyin...")
pipeline = SpotifyRecommenderPipeline()
print("✅ Sistem Hazır!")

AUDIO_FOLDER = "data/audio_snippets"

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    query = request.args.get('query', '')
    
    seed_uri = "spotify:track:0uqPG793dkDDN7sCUJJIVC" 
    
    raw_results = pipeline.recommend_hybrid([seed_uri], w_audio=0.4, w_co=0.6, top_n=3)
    
    formatted_results = []
    
    for item in raw_results:
        track_id = item['uri'].split(':')[-1]
        audio_file = f"{track_id}.mp3"
        audio_url = f"http://127.0.0.1:5000/audio/{audio_file}"
        
        formatted_results.append({
            "title": item.get('track_name', 'Bilinmeyen Şarkı'),
            "artist": item.get('artist_name', 'Bilinmeyen Sanatçı'),
            "score": str(int(item.get('score', 0) * 100)),
            "image": "https://i.scdn.co/image/ab67616d0000b273e2e352d89826aef6dbd5ff8f",
            "url": audio_url
        })
        
    return jsonify(formatted_results)

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)