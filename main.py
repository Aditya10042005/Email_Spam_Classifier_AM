import streamlit as st
import pickle
import string
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_email_msg = st.text_area('Enter the message')

if st.button("Predict"):
    transformed_email = transform_text(input_email_msg)

    vector_input = tfidf.transform([transformed_email])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")