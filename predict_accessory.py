import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys

MODEL_PATH = "models/electro_accessory_model.tflite"
IMAGE_SIZE = (224, 224)

accessory_classes = [
    'Charger',
    'Game-Controller',
    'Headphone',
    'Keyboard',
    'Laptop',
    'Monitor',
    'Mouse',
    'Smartphone',
    'Smartwatch',
    'Speaker'
]

def preprocess_image(image_path):
    """Load and preprocess the image with proper type conversion"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)  # Convert to float32
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_accessory(image_path, debug=True):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Check expected input type
    if debug:
        print(f"\nModel expects input dtype: {input_details['dtype']}")
        print(f"Model expects input shape: {input_details['shape']}")

    # Preprocess image
    input_data = preprocess_image(image_path)
    
    if debug:
        print(f"\nInput data dtype: {input_data.dtype}")
        print(f"Input data shape: {input_data.shape}")

    # Ensure input data type matches model expectations
    if input_details['dtype'] != input_data.dtype:
        input_data = input_data.astype(input_details['dtype'])
        if debug:
            print(f"Converted input to dtype: {input_data.dtype}")

    # Set input and run inference
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details['index'])[0]
    probabilities = tf.nn.softmax(output).numpy()

    # Debug output
    if debug:
        print("\nPrediction Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{accessory_classes[i]:<15}: {prob:.4f}")

    predicted_index = np.argmax(probabilities)
    return accessory_classes[predicted_index], float(probabilities[predicted_index])

if __name__ == "__main__":



    try:
        prediction, confidence = predict_accessory("test3.jpeg")
        print(f"\nFinal Prediction: {prediction} (Confidence: {confidence:.2%})")
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")