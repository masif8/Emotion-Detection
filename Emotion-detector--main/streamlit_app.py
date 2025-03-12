import streamlit as st 
import nltk
import numpy as np 
import pickle 
from nltk.stem import PorterStemmer
import re


# here we have chenge the code
nltk.download('stopwords') 


model = pickle.load(open('Logistic_regression.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))



#++++++++++++Function +++++++++++++++++++

#text Clean +++++++++++++++++++
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)






stopwords = set(nltk.corpus.stopwords.words('english'))
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label =  np.max(model.predict(input_vectorized))

    return predicted_emotion,label


# =============================APP =================================



st.title('Emotion Detection App')
st.markdown(
    """
Feeling curious about the emotions behind those words?
Paste a text snippet, and let's dive into its emotional core!
"""
)

st.subheader(['Happy','Anger','Love','Sad'])
#input_text = st.text_input('Past Your Text Here ' , height=150)
input_text = st.text_area(label="Paste Your Text Here", height=100)


#==========Prediction =============

if st.button('Predict'):
    predicted_emotion,label = predict_emotion(input_text)
    predicted_emotion = predicted_emotion.upper()
    if predicted_emotion=="JOY":
        predicted_emotion="Happy"
    
    st.write("Predicted Emotion:", predicted_emotion)
    

