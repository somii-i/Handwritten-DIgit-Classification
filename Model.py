import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import numpy as np

def cnn_model(input_shape=(28, 28, 1), num_labels=10):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_labels, activation='softmax')  # 10 output neurons for digits (0-9)
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Example: Training on MNIST dataset
def train_cnn(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values
    model = cnn_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model

# Example: Loading and training on MNIST dataset
def main():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)  # Reshape to (28, 28, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    model = train_cnn(X_train, y_train, X_test, y_test)
    model.save(r"C:\Users\ksomi\Desktop\Project\main\digit_classifier.h5")  # Save trained model

if __name__ == "__main__":
    main()
