import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# ── Groq LLM agent ──────────────────────────────────────────────────────────
import groq_agent

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))


# ═══════════════════════════════════════════════════════════════════════════
#  NEW  — REST API endpoint: CNN prediction + Groq LLM diagnosis
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a plant leaf image via multipart form-data (field name: 'file'),
    runs it through the CNN, then enriches the result with a Groq LLM
    diagnosis and returns everything as JSON.

    Example:
        curl -X POST -F "file=@leaf.jpg" http://localhost:5000/predict
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded image
    filename = file.filename
    file_path = os.path.join('static/uploads', filename)
    file.save(file_path)

    # ── CNN prediction ───────────────────────────────────────────────────
    pred_index = prediction(file_path)
    predicted_class = CNN.idx_to_classes[pred_index]

    # Compute confidence via softmax
    image = Image.open(file_path).resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    with torch.no_grad():
        logits = model(input_data)
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence = float(probs[0][pred_index])

    # ── Groq LLM diagnosis ──────────────────────────────────────────────
    llm_result = groq_agent.get_diagnosis(predicted_class, confidence)

    # ── Combined response ────────────────────────────────────────────────
    response = {
        "cnn_prediction": {
            "class": predicted_class,
            "class_index": int(pred_index),
            "confidence": round(confidence, 4),
        },
        "llm_diagnosis": llm_result,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
