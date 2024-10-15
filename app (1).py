import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Charger le modèle
model = load_model('model.h5')

# Interface utilisateur
st.title("Prédiction de Churn")

# Collecte des données de l'utilisateur pour la prédiction
age = st.number_input('Âge', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance du compte', min_value=0.0, value=50000.0)
num_of_products = st.number_input('Nombre de produits', min_value=1, max_value=4, value=1)
gender = st.selectbox('Sexe', ('Homme', 'Femme'))
geography = st.selectbox('Pays', ('France', 'Allemagne', 'Espagne'))

# Conversion du genre et du pays en valeurs numériques
gender_encoded = 0 if gender == 'Homme' else 1
geography_encoded = 1 if geography == 'France' else (2 if geography == 'Allemagne' else 0)

# Prédiction
if st.button('Prédire'):
    features = np.array([[age, balance, num_of_products, gender_encoded, geography_encoded]])
    
    # Normalisation des données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Assurez-vous que les mêmes transformations que lors de l'entraînement sont appliquées
    
    prediction = model.predict(features_scaled)
    churn_prediction = (prediction > 0.5).astype(int)
    
    if churn_prediction == 1:
        st.write("Le client est susceptible de quitter.")
    else:
        st.write("Le client est susceptible de rester.")
