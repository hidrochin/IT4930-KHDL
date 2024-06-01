import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from preprocessing import preprocess_comment
from tensorflow.keras.preprocessing.text import Tokenizer
from model.create_models import create_model_GRU, create_model_LSTM, create_model_conv

app = Flask(__name__)

EMBEDDING_DIM = 16
MAXLEN = 120  # Độ dài tối đa của chuỗi đầu vào
VOCAB_SIZE_DL = 5637 
VOCAB_SIZE_ML = 6315   
OOV_TOKEN = "<OOV>"

#Initialize models
models_name = ['Logistic Regression', 'Random Forrest', 'SVM', 'LSTM', 'GRU', 'CONV']
# models_name = [ 'LSTM', 'GRU', 'CONV']
dl = [ 'LSTM', 'GRU', 'CONV']
ml = ['Logistic Regression', 'Random Forrest', 'SVM']
model_LSTM = create_model_LSTM(VOCAB_SIZE_DL, EMBEDDING_DIM, MAXLEN)
model_GRU = create_model_GRU(VOCAB_SIZE_DL, EMBEDDING_DIM, MAXLEN)
model_CONV = create_model_conv(VOCAB_SIZE_DL, EMBEDDING_DIM, MAXLEN)

#Build models first
model_LSTM.build(input_shape=(EMBEDDING_DIM, MAXLEN))
model_GRU.build(input_shape=(None, MAXLEN))
model_CONV.build(input_shape=(None, MAXLEN))

# Load weights
model_LSTM.load_weights('model/LSTM.weights.h5')
model_GRU.load_weights('model/GRU.weights.h5')
model_CONV.load_weights('model/conv.weights.h5')

with open('model/lr_classifier.pkl', 'rb') as file:
    lr_classifier = pickle.load(file)
with open('model/rf_classifier.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)
with open('model/svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
tokenizer = Tokenizer(num_words=MAXLEN, oov_token=OOV_TOKEN)

label_map = {
    'quality': 0,
    'service': 1,
    'packing': 2,
    'shipping': 3
}
inverse_label_map = {v: k for k, v in label_map.items()}

@app.route('/')
def home():
    return render_template('index.html', models = models_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    print(data)
    new_comment = data['comment']
    selected_model = data['model']
    cleaned_comment = preprocess_comment(new_comment)

    tokenizer.fit_on_texts(cleaned_comment)
    new_comment_seq = tokenizer.texts_to_sequences([cleaned_comment])
    new_comment_padded_DL = pad_sequences(new_comment_seq, maxlen=MAXLEN, padding='post')
    new_comment_padded_ML = pad_sequences(new_comment_seq, maxlen=VOCAB_SIZE_ML, padding='post')

    predictions = {}
    for model_name, model in zip(models_name, [lr_classifier, rf_classifier, svm_classifier, model_LSTM, model_GRU, model_CONV]):
    # for model_name, model in zip(models_name, [ model_LSTM, model_GRU, model_CONV]):
        if model_name in dl:
            prediction = model.predict(new_comment_padded_DL)
            predicted_label = np.argmax(prediction, axis=0)[0]
            print(np.argmax(prediction))
        elif model_name in ml:
            prediction = model.predict(new_comment_padded_ML)
            predicted_label = np.argmax(prediction)
            print(model_name)
        predicted_label_name = inverse_label_map[predicted_label]
        predictions[model_name] = predicted_label_name
        print(prediction)
    
    return render_template('result.html', comment = new_comment, model = selected_model, result = predictions[selected_model])

if __name__ == '__main__':
    app.run(debug=True)


#Giao hàng nhanh nhưng đóng gói tệ quá