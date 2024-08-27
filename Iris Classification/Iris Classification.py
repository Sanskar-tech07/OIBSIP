import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Load iris dataset
data = load_iris()
features = data.data
target = data.target

# Split data into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
feature_train_scaled = scaler.fit_transform(feature_train)
feature_test_scaled = scaler.transform(feature_test)

# Create and train a random forest classifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(feature_train_scaled, target_train)

# Make predictions with the random forest classifier
predictions = forest.predict(feature_test_scaled)

# Calculate the accuracy of the random forest classifier
accuracy = accuracy_score(target_test, predictions)
print("Random Forest Classifier Accuracy:", accuracy)

# Create a neural network model
neural_network = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the neural network model
neural_network.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the neural network model
neural_network.fit(feature_train_scaled, target_train, epochs=50, batch_size=5, verbose=1)

# Evaluate the neural network model
loss, accuracy = neural_network.evaluate(feature_test_scaled, target_test)
print("\nNeural Network Accuracy:", accuracy)
