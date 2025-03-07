import os
from scipy.io import loadmat 
import numpy as np
from Model import cnn_model, train_cnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# Loading mat file
data = loadmat('C:\Users\ksomi\Desktop\Project\main\main\mnist-original.mat')

X = data['data'].T  # Transpose to shape (70000, 784)
y = data['label'].flatten()

# Normalize and reshape data
X = X / 255.0  # Normalize pixel values
X = X.reshape(-1, 28, 28, 1)  # Reshape to (70000, 28, 28, 1)

# Split dataset
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Load or train the CNN model
model_path = "digit_classifier.h5"
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Loaded pre-trained model.")
else:
    model = keras.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save(model_path)
    print("Model trained and saved.")

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

# Predict example digits
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)

# Precision calculation
true_positive = np.sum(pred_labels == y_test)
false_positive = len(y_test) - true_positive
precision = true_positive / (true_positive + false_positive)
print(f'Precision = {precision:.4f}')
