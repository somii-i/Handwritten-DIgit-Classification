import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import *
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageDraw

import os

# Load the trained CNN model
model = load_model("C:\Users\ksomi\Desktop\Project\main\digit_classifier.h5")
print(model.summary())

# Function to predict digit
def predict_digit(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    return np.argmax(prediction)

# Function to draw on the canvas
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x+10, y+10, fill='black', width=5)
    draw_area.line([x, y, x+10, y+10], fill='black', width=5)

# Function to clear the canvas
def clear_canvas():
    canvas.delete('all')
    draw_area.rectangle((0, 0, 300, 300), fill='white')

# Function to handle prediction
def classify():
    #filename = 'digit.png'
    #image.save(filename)
    #image_open = Image.open(filename)
    digit = predict_digit(image)
    label.config(text=f'Predicted Digit: {digit}', font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')


# GUI setup
root = Tk()
root.title("Handwritten Digit Recognition")
root.geometry("400x500")
root.configure(bg='#34495e')

label = Label(root, text="Draw a Digit", font=('Arial', 16, 'bold'), bg='#34495e', fg='white')
label.pack(pady=10)

canvas = Canvas(root, width=300, height=300, bg='white', relief='solid', borderwidth=2)
canvas.pack()

image = Image.new('RGB', (300, 300), 'white')
draw_area = ImageDraw.Draw(image)

canvas.bind('<B1-Motion>', draw)

button_frame = Frame(root, bg='#34495e')
button_frame.pack(pady=20)

predict_button = Button(button_frame, text="Predict", command=classify, font=('Arial', 14, 'bold'), bg='#27ae60', fg='white', padx=10, pady=5)
predict_button.grid(row=0, column=0, padx=10)

clear_button = Button(button_frame, text="Clear", command=clear_canvas, font=('Arial', 14, 'bold'), bg='#c0392b', fg='white', padx=10, pady=5)
clear_button.grid(row=0, column=1, padx=10)

root.mainloop()

