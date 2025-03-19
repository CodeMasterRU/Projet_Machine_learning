import joblib
import streamlit as st
import pandas as pd
import spacy

try:
    logistic_model_rating = joblib.load('Logistic Regression_rating_model.pkl')
    rf_model_rating = joblib.load('Random Forest_rating_model.pkl')
    logistic_model_rec = joblib.load('Logistic Regression_rec_model.pkl')
    rf_model_rec = joblib.load('Random Forest_rec_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading models or vectorizer: {e}")
    st.stop()

def check_model_fitted(model):
    try:
        # Проверяем, можно ли сделать предсказание (например, с пустым массивом)
        model.predict([[0]])
        return True
    except Exception as e:
        return False
models = {
    'Logistic Regression (Rating)': logistic_model_rating,
    'Random Forest (Rating)': rf_model_rating,
    'Logistic Regression (Recommended)': logistic_model_rec,
    'Random Forest (Recommended)': rf_model_rec
}

for model_name, model in models.items():
    if not check_model_fitted(model):
        st.warning(f"Model '{model_name}' is not fitted yet.")

nlp = spacy.load('en_core_web_sm')

st.title('Text Prediction App')

st.sidebar.title('Prediction Options')
prediction_type = st.sidebar.selectbox('Select Prediction Type:', ['Rating', 'Recommended IND'])

user_input = st.text_area("Enter the text for prediction:")

def process_text(text):
    if pd.isna(text):
        return ""
    doc = nlp(text)
    
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space]
    
    return " ".join(tokens)

def predict_review(review_text):
    processed = process_text(review_text)
    
    features = tfidf_vectorizer.transform([processed])
    
    rating = logistic_model_rating.predict(features)[0]
    recommended = logistic_model_rec.predict(features)[0]
    
    return {
        'Rating': rating,
        'Recommended': 'Yes' if recommended == 1 else 'No'
    }

if user_input:
    predictions = predict_review(user_input)
    
    st.write(f"Predicted Rating: {predictions['Rating']}")
    st.write(f"Recommended IND Prediction: {predictions['Recommended']}")
    
else:
    st.warning("Please enter text for prediction.")
