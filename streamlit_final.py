import streamlit as st
import joblib
import spacy

rating_model = joblib.load("models/rating_model.pkl")
rec_model = joblib.load("models/recommendation_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def predict_review(review_text):
    processed = process_text(review_text)
    vector = vectorizer.transform([processed])
    rating = rating_model.predict(vector)[0]
    recommended = rec_model.predict(vector)[0]
    return round(rating, 1), "Recommended" if recommended == 1 else "Not Recommended"

st.title("🛍️ Prédire les notes et les recommandations des produits")

st.write("Saisissez un avis et le modèle prédira une note (de 1 à 5) et la probabilité de recommandation.")

review_text = st.text_area("Saisissez le texte de votre avis:", "")

if st.button("Prédire"):
    if review_text.strip():
        rating, recommendation = predict_review(review_text)
        st.success(f"**Note prévue :** {rating}")
        st.info(f"**Recommandation:** {recommendation}")
    else:
        st.warning("Saisissez le texte de votre avis!")

