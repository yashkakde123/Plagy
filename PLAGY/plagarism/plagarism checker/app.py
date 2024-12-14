from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import PyPDF2
import io

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    probabilities = model.predict_proba(vectorized_text)
    plagiarism_probability = probabilities[0][1]  # Probability of class 1 (Plagiarism)
    plagiarism_percentage = plagiarism_probability * 100
    
    return f"Plagiarism Detected: {plagiarism_percentage:.2f}% plagarized" if plagiarism_probability >= 0.5 else f"No Plagiarism: {100 - plagiarism_percentage:.2f}% original"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    input_text = ""
    if file.filename.endswith('.txt'):
        input_text = file.read().decode('utf-8')  # Read text file directly
    elif file.filename.endswith('.pdf'):
        with io.BytesIO(file.read()) as f:  # Read PDF file directly into memory
            reader = PyPDF2.PdfReader(f)
            input_text = ""
            for page in reader.pages:
                input_text += page.extract_text()  # Extract text from each page
    
    # Call the detect function with the file contents
    detection_result = detect(input_text)
    
    return render_template('index.html', result=detection_result)  # Pass the result to the template

if __name__ == "__main__":
    app.run(debug=True)
