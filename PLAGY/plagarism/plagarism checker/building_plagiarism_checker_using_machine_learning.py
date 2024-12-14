


import nltk
nltk.download("popular")
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
from nltk.corpus import stopwords
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

"""# Load Dataset"""

data = pd.read_csv("C:/Users/ASUS/Desktop/PLAGY/plagarism/plagarism checker/dataset.csv")
data.head()

data['label'].value_counts()

"""# Clean Text"""

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text
data["source_text"] = data["source_text"].apply(preprocess_text)
data["plagiarized_text"] = data["plagiarized_text"].apply(preprocess_text)

"""# Vectorization"""

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data["source_text"] + " " + data["plagiarized_text"])

y = data["label"]

"""# Train Test Split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# applying logistic regression"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix")
print(cm)

"""# Random Forest Model"""

from sklearn.ensemble import RandomForestClassifier
# Instantiate the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate classification report
classification_rep = classification_report(y_test, y_pred)
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Print results
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(cm)

"""# Naiv Bays Model"""

from sklearn.naive_bayes import MultinomialNB
# Instantiate the model
model = MultinomialNB()
# Fit the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate classification report
classification_rep = classification_report(y_test, y_pred)
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Print results
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(cm)

"""# SVM"""

from sklearn.svm import SVC

# Instantiate the model with probability=True
model = SVC(kernel='linear', random_state=42, probability=True)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
classification_rep = classification_report(y_test, y_pred)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(cm)

"""# Save SVM and Vectorizor"""

import pickle

pickle.dump(model,open("model.pkl",'wb'))
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl','wb'))

"""# Load Model and Vectorizer"""

model = pickle.load(open('model.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

"""# Detection System"""

def detect(input_text):
    # Vectorize the input text
    vectorized_text = tfidf_vectorizer.transform([input_text])
    
    # Get the predicted probabilities for each class
    probabilities = model.predict_proba(vectorized_text)
    
    # Assuming class 1 is "Plagiarism" and class 0 is "No Plagiarism"
    plagiarism_probability = probabilities[0][1]  # Probability of class 1 (Plagiarism)
    
    # Convert to percentage
    plagiarism_percentage = plagiarism_probability * 100
    
    return f"Plagiarism Detected: {plagiarism_percentage:.2f}%" if plagiarism_probability >= 0.5 else f"No Plagiarism: {100 - plagiarism_percentage:.2f}%"

# example ( it is a plagarized text)
input_text = 'Researchers have discovered a new species of butterfly in the Amazon rainforest.'
detect(input_text)

# example ( it has no plagiarism)
input_text = 'Playing musical instruments enhances creativity.'
detect(input_text)

# example ( it has no plagarism)
input_text = 'Practicing yoga enhances physical flexibility.'
detect(input_text)

# sklearn version
import sklearn
sklearn.__version__

