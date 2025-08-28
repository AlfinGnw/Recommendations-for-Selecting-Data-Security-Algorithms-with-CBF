from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
import pickle
import os

app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Fungsi preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words(['english','indonesian']))
    return ' '.join([word for word in tokens if word not in stop_words])

# Load data dan model
def load_data():
    with open('dataset.json', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['processed'] = (df['Judul'] + ' ' + df['Abstrak'] + ' ' + 
                      df['algoritma'] + ' ' + df['tipe_data']).apply(preprocess)
    
    # Train atau load model
    if not os.path.exists('model.pkl'):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['processed'])
        with open('model.pkl', 'wb') as f:
            pickle.dump((vectorizer, tfidf_matrix), f)
    else:
        with open('model.pkl', 'rb') as f:
            vectorizer, tfidf_matrix = pickle.load(f)
    
    return df, vectorizer, tfidf_matrix

# Route utama
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        df, vectorizer, tfidf_matrix = load_data()
        
        # Proses query
        processed_query = preprocess(query)
        query_vector = vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        df['similarity'] = similarities
        
        # Ambil 3 teratas
        recommendations = df.sort_values('similarity', ascending=False).head(3)
        return render_template('result.html', 
                            query=query,
                            recommendations=recommendations.to_dict('records'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)