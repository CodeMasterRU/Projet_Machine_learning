# Enregistrez dans le fichier streamlit.py
import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np

# Chargement des modèles sauvegardés
@st.cache_resource
def load_models():
    model = joblib.load('./models/recommendation_model.pkl')
    reg_model = joblib.load('./models/rating_model.pkl')
    tfidf = joblib.load('./models/tfidf_vectorizer.pkl')
    return model, reg_model, tfidf

model, reg_model, tfidf = load_models()

# Fonction pour prédire la recommandation basée sur le texte de l'avis
def predict_recommendation(review_text, age, rating):
    # Transformation du texte
    text_features = tfidf.transform([review_text])
    text_df = pd.DataFrame(text_features.toarray(), columns=tfidf.get_feature_names_out())
    
    # Création des caractéristiques numériques
    numeric_data = pd.DataFrame({
        'Unnamed: 0': [0],
        'Clothing ID': [0],
        'Age': [age],
        'Rating': [rating],
        'Positive Feedback Count': [0]
    })
    
    # Obtention de la liste des colonnes du modèle
    # Supposons que le modèle a un attribut feature_names_in_ ou n_features_in_
    try:
        # Tentative d'obtenir la liste des colonnes du modèle
        if hasattr(model, 'feature_names_in_'):
            columns = model.feature_names_in_
        else:
            # Si l'attribut feature_names_in_ n'existe pas, créons une liste de base
            columns = list(numeric_data.columns) + list(text_df.columns)
    except:
        # Si nous ne pouvons pas obtenir la liste des colonnes, utilisons une liste de base
        columns = list(numeric_data.columns) + list(text_df.columns)
    
    # Préparation du dataframe complet des caractéristiques
    full_features = pd.DataFrame(0, index=[0], columns=columns)
    
    # Remplissage des caractéristiques numériques
    for col in numeric_data.columns:
        if col in full_features.columns:
            full_features[col] = numeric_data[col].values
    
    # Remplissage des caractéristiques textuelles
    for col in text_df.columns:
        if col in full_features.columns:
            full_features[col] = text_df[col].values
    
    # Prédiction
    prediction = model.predict(full_features)
    probability = model.predict_proba(full_features)
    
    return {
        'recommendation': 'Recommandé' if prediction[0] == 1 else 'Non recommandé',
        'confidence': probability[0][1] if prediction[0] == 1 else probability[0][0]
    }

st.title('Analyseur d\'avis sur les vêtements pour femmes')

# Formulaire pour saisir un avis
st.header('Entrez un avis pour analyse')
review_text = st.text_area('Texte de l\'avis', height=150)
age = st.number_input('Âge', min_value=0, max_value=100, value=30)
rating = st.number_input('Note', min_value=1, max_value=5, value=4)

if st.button('Analyser'):
    if review_text:
        # Analyse de la tonalité
        sentiment = TextBlob(review_text).sentiment.polarity
        st.write(f'Tonalité de l\'avis: {sentiment:.2f} (-1 négatif, 1 positif)')
        
        # Prédiction de la recommandation
        result = predict_recommendation(review_text, age, rating)
        st.write(f'Prédiction: {result["recommendation"]} (confiance: {result["confidence"]:.2f})')
        
        # Visualisation des résultats
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Diagramme de tonalité
        ax[0].bar(['Négatif', 'Neutre', 'Positif'], 
                 [max(0, -sentiment), 1-abs(sentiment), max(0, sentiment)])
        ax[0].set_title('Répartition de la tonalité')
        
        # Diagramme de confiance dans la recommandation
        ax[1].bar(['Non recommandé', 'Recommandé'], 
                 [1-result["confidence"], result["confidence"]])
        ax[1].set_title('Confiance dans la recommandation')
        
        st.pyplot(fig)
    else:
        st.error('Veuillez entrer un texte d\'avis.')