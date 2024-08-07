import streamlit as st
import joblib

# Load the SVM model pipeline which includes the vectorizer
model_path = r"C:\Users\hp\Desktop\Team Bravo Project\Jupiter Notebook (Sentiment Analysis)\TBsentiment_mod.pkl"
svm_model = joblib.load(model_path)

# Define a function to make predictions
def predict_sentiment(text):
    return svm_model.predict([text])[0]

# Streamlit app
st.title('Sentiment Analysis App')

text = st.text_area('Enter text for sentiment analysis')

if st.button('Analyze'):
    if text:
        sentiment = predict_sentiment(text)
        st.write(f'Sentiment: {sentiment}')
    else:
        st.write('Please enter some text to analyze.')

# Run the app using the command: streamlit run app.py
