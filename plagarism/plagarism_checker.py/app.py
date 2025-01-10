import streamlit as st
import pickle
import io
import PyPDF2
from docx import Document  # For handling Word files

# Load the model and vectorizer
model = pickle.load(open('C:/Users/ASUS/Desktop/PLAGY/plagarism/plagarism_checker.py/model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('C:/Users/ASUS/Desktop/PLAGY/plagarism/plagarism_checker.py/tfidf_vectorizer.pkl', 'rb'))

# Function to detect plagiarism
def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    probabilities = model.predict_proba(vectorized_text)
    plagiarism_probability = probabilities[0][1]  # Probability of class 1 (Plagiarism)
    plagiarism_percentage = plagiarism_probability * 100
    
    return f"Plagiarism Detected: {plagiarism_percentage:.2f}% plagiarized" if plagiarism_probability >= 0.5 else f"No Plagiarism: {100 - plagiarism_percentage:.2f}% original"

# Streamlit App
st.title("Plagiarism Detection App")
st.write("Upload a file or enter text to check for plagiarism.")

# Text input
input_text = st.text_area("Enter text here:", placeholder="Paste your text here...")

# File upload
uploaded_file = st.file_uploader("Or upload a file:", type=["txt", "pdf", "docx"])

# Display uploaded file content
if uploaded_file is not None:
    st.subheader("Uploaded File Content:")
    if uploaded_file.name.endswith('.txt'):
        input_text = uploaded_file.read().decode('utf-8')  # Read text file
    elif uploaded_file.name.endswith('.pdf'):
        with io.BytesIO(uploaded_file.read()) as f:
            reader = PyPDF2.PdfReader(f)
            input_text = ""
            for page in reader.pages:
                input_text += page.extract_text()  # Extract text from PDF
    elif uploaded_file.name.endswith('.docx'):
        with io.BytesIO(uploaded_file.read()) as f:
            doc = Document(f)
            input_text = ""
            for para in doc.paragraphs:
                input_text += para.text + "\n"  # Extract text from Word file
    st.write(input_text)

# Check for plagiarism
if st.button("Check for Plagiarism"):
    if input_text:
        result = detect(input_text)  # Call the detect function
        st.subheader("Result:")
        st.success(result)
    else:
        st.error("Please enter text or upload a file to check for plagiarism.")