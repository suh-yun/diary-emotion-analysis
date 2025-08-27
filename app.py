from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os, random

# ---- 경로 설정 ----
BASE_DIR = '/home/pi/emotion_project'  # 라즈베리파이 기준
MODEL_PATH = os.path.join(BASE_DIR, 'models/emotion_model_paragraph.tflite')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models/tokenizer_paragraph.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'models/label_encoder_paragraph.pkl')
STICKERS_DIR = os.path.join(BASE_DIR, 'stickers')
MAX_LEN = 256  # 학습 시 사용한 길이와 동일

# ---- 모델/토크나이저/라벨 로드 ----
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__, static_folder="stickers", template_folder="templates")

# ---- 감정 예측 함수 ----
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    interpreter.set_tensor(input_details[0]['index'], padded.astype(np.float32))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
    return emotion

# ---- 스티커 추천 함수 ----
def recommend_sticker(emotion):
    folder = os.path.join(STICKERS_DIR, emotion)
    if not os.path.exists(folder):
        return None
    stickers = os.listdir(folder)
    if not stickers:
        return None
    return os.path.join(folder, random.choice(stickers))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    emotion = predict_emotion(text)
    sticker_path = recommend_sticker(emotion)
    sticker_url = f"/{sticker_path}" if sticker_path else ""

    return jsonify({'emotion': emotion, 'sticker': sticker_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
