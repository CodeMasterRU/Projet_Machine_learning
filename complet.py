# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import spacy
from tqdm.notebook import tqdm

print("Missing values by column:")
print(df.isnull().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

print(f"Original dataset shape: {df.shape}")
df = df.dropna(subset=['Review Text'])
df = df.drop_duplicates()
print(f"Cleaned dataset shape: {df.shape}")

nlp = spacy.load('en_core_web_sm')

def process_text(text):
    if pd.isna(text):
        return ""
    doc = nlp(text)
    
    # Extract lemmatized tokens (excluding stopwords and punctuation)
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space]
    
    return " ".join(tokens)

sample_review = df['Review Text'].iloc[0]
print("Original review:\n", sample_review)
print("\nProcessed review:\n", process_text(sample_review))

print("Processing reviews with spaCy...")
tqdm.pandas()
df['processed_text'] = df['Review Text'].progress_apply(process_text)
print("Processing complete!")


df[['Review Text', 'processed_text']].head()

tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

print(f"TF-IDF matrix shape: {X_tfidf.shape}")
print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")

X = X_tfidf
y_rating = df['Rating']
y_recommended = df['Recommended IND']

X_train, X_test, y_rating_train, y_rating_test, y_rec_train, y_rec_test = train_test_split(
    X, y_rating, y_recommended, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

rating_models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

rating_results = {}

for name, model in rating_models.items():
    print(f"\nTraining {name} for Rating prediction...")
    
    model.fit(X_train, y_rating_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_rating_test, y_pred)
    precision = precision_score(y_rating_test, y_pred, average='weighted')
    recall = recall_score(y_rating_test, y_pred, average='weighted')
    f1 = f1_score(y_rating_test, y_pred, average='weighted')
    
    cv_scores = cross_val_score(model, X, y_rating, cv=5, scoring='accuracy')
    
    rating_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'model': model,
        'predictions': y_pred
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    print("\nClassification Report:")
    print(classification_report(y_rating_test, y_pred))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_rating_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y_rating.unique()), 
                yticklabels=sorted(y_rating.unique()))
    plt.title(f'Confusion Matrix - {name} (Rating)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

rec_models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}


rec_results = {}

for name, model in rec_models.items():
    print(f"\nTraining {name} for Recommendation prediction...")

    model.fit(X_train, y_rec_train)
    
    y_pred = model.predict(X_test)
    

    accuracy = accuracy_score(y_rec_test, y_pred)
    precision = precision_score(y_rec_test, y_pred)
    recall = recall_score(y_rec_test, y_pred)
    f1 = f1_score(y_rec_test, y_pred)
    
    cv_scores = cross_val_score(model, X, y_recommended, cv=5, scoring='accuracy')
    
    rec_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'model': model,
        'predictions': y_pred
    }

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    

    print("\nClassification Report:")
    print(classification_report(y_rec_test, y_pred))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_rec_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Recommended', 'Recommended'], 
                yticklabels=['Not Recommended', 'Recommended'])
    plt.title(f'Confusion Matrix - {name} (Recommendation)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Compare Rating models
rating_comparison = pd.DataFrame({
    'Model': list(rating_results.keys()),
    'Accuracy': [results['accuracy'] for results in rating_results.values()],
    'Precision': [results['precision'] for results in rating_results.values()],
    'Recall': [results['recall'] for results in rating_results.values()],
    'F1 Score': [results['f1'] for results in rating_results.values()],
    'CV Accuracy': [results['cv_mean'] for results in rating_results.values()]
})

print("Rating Model Comparison:")
rating_comparison.set_index('Model')

# Compare Recommendation models
rec_comparison = pd.DataFrame({
    'Model': list(rec_results.keys()),
    'Accuracy': [results['accuracy'] for results in rec_results.values()],
    'Precision': [results['precision'] for results in rec_results.values()],
    'Recall': [results['recall'] for results in rec_results.values()],
    'F1 Score': [results['f1'] for results in rec_results.values()],
    'CV Accuracy': [results['cv_mean'] for results in rec_results.values()]
})

print("Recommendation Model Comparison:")
rec_comparison.set_index('Model')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Accuracy']

plt.figure(figsize=(12, 6))
rating_comparison_melted = pd.melt(rating_comparison, id_vars=['Model'], value_vars=metrics, 
                                 var_name='Metric', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='Model', data=rating_comparison_melted)
plt.title('Rating Models Performance Comparison')
plt.ylim(0, 1)
plt.legend(title='Model')
plt.show()


plt.figure(figsize=(12, 6))
rec_comparison_melted = pd.melt(rec_comparison, id_vars=['Model'], value_vars=metrics, 
                              var_name='Metric', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='Model', data=rec_comparison_melted)
plt.title('Recommendation Models Performance Comparison')
plt.ylim(0, 1)
plt.legend(title='Model')
plt.show()

feature_names = tfidf_vectorizer.get_feature_names_out()

rating_rf = rating_results['Random Forest']['model']
rec_rf = rec_results['Random Forest']['model']

# Rating feature importance
rating_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rating_rf.feature_importances_
}).sort_values('importance', ascending=False)

# Recommendation feature importance
rec_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rec_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=rating_importance.head(20))
plt.title('Top 20 Features for Rating Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=rec_importance.head(20))
plt.title('Top 20 Features for Recommendation Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

best_rating_model = max(rating_results.items(), key=lambda x: x[1]['f1'])[1]['model']
best_rec_model = max(rec_results.items(), key=lambda x: x[1]['f1'])[1]['model']


def predict_review(review_text):
    processed = process_text(review_text)

    features = tfidf_vectorizer.transform([processed])
    
    rating = best_rating_model.predict(features)[0]
    recommended = best_rec_model.predict(features)[0]
    
    return {
        'Rating': rating,
        'Recommended': 'Yes' if recommended == 1 else 'No'
    }

test_reviews = [
    "I absolutely love this dress! The fit is perfect and the material is high quality. Definitely worth the price.",
    "The shirt was okay, but the material was a bit thin. It fits as expected though.",
    "Terrible product. Fell apart after one wash. Complete waste of money. Do not buy!"
]

for i, review in enumerate(test_reviews):
    result = predict_review(review)
    print(f"Review {i+1}: {review[:50]}...")
    print(f"Predicted Rating: {result['Rating']}")
    print(f"Predicted Recommendation: {result['Recommended']}\n")

print("RATING PREDICTION:")
print(rating_comparison.set_index('Model'))
print("\nRECOMMENDATION PREDICTION:")
print(rec_comparison.set_index('Model'))

best_rating_model_name = max(rating_results.items(), key=lambda x: x[1]['f1'])[0]
best_rec_model_name = max(rec_results.items(), key=lambda x: x[1]['f1'])[0]

print(f"\nBest model for Rating prediction: {best_rating_model_name}")
print(f"Best model for Recommendation prediction: {best_rec_model_name}")

print("\nTop 10 most important words for Rating prediction:")
print(rating_importance.head(10))

print("\nTop 10 most important words for Recommendation prediction:")
print(rec_importance.head(10))