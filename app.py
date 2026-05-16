import numpy as np
from flask import Flask, request, render_template
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
import nltk
import pickle

app = Flask(__name__)

# Load saved model and transformers
clf = pickle.load(open("multinomial_nb_model.pkl", "rb"))
count_vect = pickle.load(open("count_vect.pkl", "rb"))
tfidf_transformer = pickle.load(open("tfidf_transformer.pkl", "rb"))

twenty_train = fetch_20newsgroups(subset='test', shuffle=True)
target_names = twenty_train.target_names
stemmer = PorterStemmer()

# Download punkt once (before request handling)
nltk.download("punkt")
nltk.download("punkt_tab")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Stem input documents
    def stem_documents(docs):
        return [
            " ".join([stemmer.stem(word) for word in word_tokenize(doc)])
            for doc in docs
        ]

    input_text = [x for x in request.form.values()]
    stemmed_data = stem_documents(input_text)

    # Transform input
    x_train_counts = count_vect.transform(stemmed_data)
    x_train_tfidf = tfidf_transformer.transform(x_train_counts)

    # Predict
    y = clf.predict(x_train_tfidf)
    prediction = target_names[y[0]]

    return render_template(
        'index.html',
        prediction_text=f'The text status is {prediction.split(".")[-1].capitalize()}'
    )

if __name__ == "_main_":
    app.run(debug=True)