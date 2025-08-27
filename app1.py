from flask import Flask, request, jsonify
import os, random

BASE_DIR = '/home/eei2024/diary-emotion-analysis'
STICKERS_DIR = os.path.join(BASE_DIR, 'stickers')

app = Flask(__name__, static_folder="stickers")

def predict_emotion(text):
    return random.choice(['happy', 'sad', 'neutral'])

def recommend_sticker(emotion):
    folder = os.path.join(STICKERS_DIR, emotion)
    if not os.path.exists(folder):
        return None
    stickers = os.listdir(folder)
    if not stickers:
        return None
    return f"/stickers/{emotion}/{random.choice(stickers)}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    emotion = predict_emotion(text)
    sticker_url = recommend_sticker(emotion)

    return jsonify({
        'emotion': emotion,
        'sticker': sticker_url
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
