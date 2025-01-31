# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found. Ensure model.pkl and tfidf_vectorizer.pkl are in the root directory.")
    exit(1)

@app.route('/api/check-plagiarism', methods=['POST'])
def check_plagiarism():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess and vectorize the input text
        vectorized_text = tfidf_vectorizer.transform([data['text']])
        
        # Make prediction
        prediction = model.predict(vectorized_text)
        probability = model.predict_proba(vectorized_text)[0][1]  # Probability of plagiarism
        
        return jsonify({
            'isPlagiarized': bool(prediction[0]),
            'confidence': float(probability),
            'message': 'Plagiarism Detected' if prediction[0] == 1 else 'No Plagiarism Detected'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Service is running'})

if __name__ == "__main__":
    app.run(debug=True, port=5000)