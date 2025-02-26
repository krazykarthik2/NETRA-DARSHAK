import cv2
import torch
import os
import easyocr
import threading
import time
from flask import Flask, render_template_string, Response, request, jsonify
from gtts import gTTS
from playsound import playsound
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator

# Flask app setup
app = Flask(__name__)

# AI Model Setup (Scene Description)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

translator = Translator()
reader = easyocr.Reader(['en'])  # OCR model
selected_language = 'en'  # Default language

LANGUAGE_CODES = {
    "english": "en", "telugu": "te", "hindi": "hi", "tamil": "ta", "kannada": "kn",
    "marathi": "mr", "bengali": "bn", "gujarati": "gu", "malayalam": "ml", "urdu": "ur"
}

# Camera Setup
camera = cv2.VideoCapture(0)

def generate_frames():
    """Continuously capture frames and send them to the web app."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the web page."""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Netra Darshak</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f4; }
            h1 { color: #333; }
            select, button { padding: 10px; margin: 10px; }
            #output { margin-top: 20px; font-size: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Netra Darshak - AI Vision Assistant</h1>
        <label for="language">Select Language:</label>
        <select id="language">
            <option value="en">English</option>
            <option value="te">Telugu</option>
            <option value="hi">Hindi</option>
            <option value="ta">Tamil</option>
        </select>
        <button onclick="startDescription()">Start Description</button>
        <br><img src="/video_feed" width="640" height="480">
        <p id="output">Scene Description: </p>
        <script>
            function startDescription() {
                let language = document.getElementById('language').value;
                fetch('/set_language', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify({ language: language }) 
                });
                setInterval(() => {
                    fetch('/describe')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('output').innerText = "Scene Description: " + data.translated_caption;
                    });
                }, 5000);
            }
        </script>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_language', methods=['POST'])
def set_language():
    global selected_language
    lang = request.json.get("language", "en")
    selected_language = lang
    return jsonify({"message": f"Language set to {lang}"})

@app.route('/describe')
def describe_scene():
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"})
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    translated_caption = translator.translate(caption, dest=selected_language).text
    speak_text(translated_caption)
    return jsonify({"caption": caption, "translated_caption": translated_caption})

def speak_text(text):
    try:
        tts = gTTS(text=text, lang=selected_language)
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        print("Speech error:", e)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
