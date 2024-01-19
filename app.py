import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

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

# Load your saved models and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set the background color
bg_color = "linear-gradient(135deg, #0074E4 0%, #00F260 100%)"  # Example gradient colors
st.markdown(f"""
    <style>
        .reportview-container {{
            background: {bg_color};
        }}
    </style>
""", unsafe_allow_html=True)

st.title("Fake Product Review Monitoring System")

input_sms = st.text_area("Enter a review to detect ham/spam")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown("<h2 style='color: red;'>Spam</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>Ham</h2>", unsafe_allow_html=True)
