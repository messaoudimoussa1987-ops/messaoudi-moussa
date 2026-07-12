import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. Création d'un jeu de données fictif (Simulation)
# Supposons que nous avons 4 capteurs : 
# [Température moteur (°C), Température extérieure (°C), Vitesse (km/h), Niveau liquide de refroidissement (%)]

# Générer des données normales (Classe 0 : Pas de surchauffe)
# Moteur: 80-95°C, Ext: 30-45°C (Oran en juillet), Vitesse: 20-60, Liquide: 70-100%
normal_data = np.random.rand(500, 4) * [15, 15, 40, 30] + [80, 30, 20, 70]
normal_labels = np.zeros(500)

# Générer des données de surchauffe (Classe 1 : Risque de panne)
# Moteur: 100-120°C, Ext: 35-45°C, Vitesse: 0-30, Liquide: 10-50%
overheat_data = np.random.rand(500, 4) * [20, 10, 30, 40] + [100, 35, 0, 10]
overheat_labels = np.ones(500)

# Combiner et préparer les données
X = np.vstack((normal_data, overheat_data))
y = np.concatenate((normal_labels, overheat_labels))

# Séparation en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Construction du Réseau de Neurones avec Keras
model = Sequential([
    # Couche d'entrée et première couche cachée (4 entrées pour nos 4 capteurs)
    Dense(16, activation='relu', input_shape=(4,)),
    # Deuxième couche cachée
    Dense(8, activation='relu'),
    # Couche de sortie (1 neurone avec Sigmoid pour une classification binaire 0 ou 1)
    Dense(1, activation='sigmoid')
])

# 3. Compilation du modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Entraînement du modèle
print("Début de l'entraînement du modèle...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 5. Évaluation du modèle
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrécision du modèle sur les données de test : {accuracy * 100:.2f}%")

# 6. Test avec de nouvelles données (Un bus sur la route d'Es-Sénia)
# Exemple : Temp Moteur 115°C, Temp Extérieure 42°C, Vitesse 15 km/h, Liquide 25%
bus_es_senia = np.array([[115.0, 42.0, 15.0, 25.0]])

prediction = model.predict(bus_es_senia)
risque = prediction[0][0] * 100

print("\n--- Analyse en temps réel ---")
print(f"Probabilité de surchauffe imminente : {risque:.1f}%")
if risque > 50:
    print("ALERTE : Risque de panne détecté. Veuillez arrêter le bus.")
else:
    print("STATUT : Le bus fonctionne normalement.")
