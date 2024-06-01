from flask import Flask, request, render_template
import pickle
from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from preprocessing import preprocess_comment  # Import the preprocessing function

app = Flask(__name__)

# Constants and label mappings
MAXLEN = 120  # Độ dài tối đa của chuỗi đầu vào
label_map = {
    'quality': 0,
    'service': 1,
    'packing': 2,
    'shipping': 3
}
inverse_label_map = {v: k for k, v in label_map.items()}

# Load machine learning models
ml_models = {
    'model1': pickle.load(open('models/lr_classifier.pkl', 'rb')),
    'model2': pickle.load(open('models/rf_classifier.pkl', 'rb')),
    'model3': pickle.load(open('models/svm_classifier.pkl', 'rb'))
}

# Load deep learning models
dl_models = {
    'model4': load_model('image/LSTM.weights.h5'),
    'model5': load_model('image/GRU.weights.h5'),
    'model6': load_model('image/conv.weights.h5')
}

# Tokenizer for text preprocessing
tokenizer = Tokenizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment']
        model_name = request.form['model']
        result = predict(comment, model_name)
        return render_template('index.html', result=result, comment=comment, model_name=model_name)
    return render_template('index.html')

def preprocess(comment):
    # Use the preprocessing function from preprocessing.py
    processed_comment = preprocess_comment(comment)
    tokenizer.fit_on_texts([processed_comment])
    sequences = tokenizer.texts_to_sequences([processed_comment])
    padded_sequences = pad_sequences(sequences, maxlen=MAXLEN)
    return padded_sequences

def predict(comment, model_name):
    processed_comment = preprocess(comment)
    if model_name in ml_models:
        model = ml_models[model_name]
        prediction = model.predict([processed_comment])
        label = inverse_label_map[np.argmax(prediction)]
    elif model_name in dl_models:
        model = dl_models[model_name]
        prediction = model.predict(processed_comment)
        label = inverse_label_map[np.argmax(prediction)]
    return label

if __name__ == '__main__':
    app.run(debug=True)
