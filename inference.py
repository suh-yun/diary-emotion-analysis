import tensorflow as tf
import numpy as np
import pickle

MODEL_PATH = "emotion_model_paragraph.tflite"
TOKENIZER_PATH = "tokenizer_paragraph.pkl"
LABEL_ENCODER_PATH = "label_encoder_paragraph.pkl"

input_text = "It's a good day!"

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 256
sequence = tokenizer.texts_to_sequences([input_text])
padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
interpreter.set_tensor(input_details[0]['index'], np.array(padded_sequence, dtype=np.float32))
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Decode prediction
predicted_class = np.argmax(output)
predicted_label = label_encoder.inverse_transform([predicted_class])[0]

print(f"input text: {input_text}")
print(f"predicted label: {predicted_label}")

