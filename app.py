from flask import Flask, request, jsonify
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from preprocessing import preprocess_comment
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load model
model_LSTM = load_model('model/LSTM.weights.h5')
model_GRU = load_model('model/GRU.weights.h5')
model_CONV = load_model('model/conv.weights.h5')

MAXLEN = 120  # Độ dài tối đa của chuỗi đầu vào

label_map = {
    'quality': 0,
    'service': 1,
    'packing': 2,
    'shipping': 3
}
inverse_label_map = {v: k for k, v in label_map.items()}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_comment = data['comment']
    cleaned_comment = preprocess_comment(new_comment)

    new_comment_seq = tokenizer.texts_to_sequences([cleaned_comment])
    new_comment_padded = pad_sequences(new_comment_seq, maxlen=MAXLEN, padding='post')

    predictions = {}
    for model_name, model in zip(label_map.keys(), [model_LSTM, model_GRU, model_CONV]):
        prediction = model.predict(new_comment_padded)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_label_name = inverse_label_map[predicted_label]
        predictions[model_name] = predicted_label_name
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
