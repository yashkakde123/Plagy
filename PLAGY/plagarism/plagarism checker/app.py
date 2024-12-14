from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import PyPDF2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    probabilities = model.predict_proba(vectorized_text)
    plagiarism_probability = probabilities[0][1]  # Probability of class 1 (Plagiarism)
    plagiarism_percentage = plagiarism_probability * 100
    
    return f"Plagiarism Detected: {plagiarism_percentage:.2f}%" if plagiarism_probability >= 0.5 else f"No Plagiarism: {100 - plagiarism_percentage:.2f}%"

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
    
    # Save the file to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Read the contents of the file
    input_text = ""
    if file.filename.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            input_text = f.read()
    elif file.filename.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            input_text = ""
            for page in reader.pages:
                input_text += page.extract_text()  # Extract text from each page
    
    # Call the detect function with the file contents
    detection_result = detect(input_text)
    
    return render_template('index.html', result=detection_result)  # Pass the result to the template


if __name__ == "__main__":
    app.run(debug=True)
