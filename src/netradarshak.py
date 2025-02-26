import cv2
import torch
import threading
import time
import os
import speech_recognition as sr
import easyocr
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from playsound import playsound
from googletrans import Translator

def check_pi_camera_stream(pi_url):
    """Check if the Raspberry Pi Zero camera stream is available."""
    try:
        response = requests.get(pi_url, timeout=3)
        if response.status_code == 200:
            print("Pi Zero camera stream detected.")
            return True
    except requests.RequestException:
        pass
    print("Pi Zero camera stream not available. Switching to laptop camera.")
    return False

def initialize_camera():
    """Initialize the camera based on availability."""
    PI_CAMERA_URL = "http://<Pi_IP>:<port>/stream"  # Replace with actual Pi stream URL
    use_pi_camera = check_pi_camera_stream(PI_CAMERA_URL)
    
    if use_pi_camera:
        cap = cv2.VideoCapture(PI_CAMERA_URL)  # Connect to Pi Zero camera
    else:
        cap = cv2.VideoCapture(0)  # Use laptop camera
    
    if not cap.isOpened():
        print("Error: Could not access the selected camera.")
        return None
    return cap

def play_prompt(message, lang='en'):
    """Generates speech from text and plays the audio."""
    try:
        tts = gTTS(text=message, lang=lang)
        audio_file = "prompt_audio.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        print("Error playing prompt:", e)

def recognize_language():
    """Uses speech recognition to determine the user's desired language."""
    prompt_message = "Please say the language you want the description in. For example, English, Telugu, or Hindi."
    play_prompt(prompt_message, lang='en')
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for language selection...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            detected_language = recognizer.recognize_google(audio).lower()
            print(f"You said: {detected_language}")
            return detected_language
        except sr.UnknownValueError:
            print("Could not understand audio, defaulting to English.")
            return "english"
        except sr.RequestError:
            print("Speech Recognition service error, defaulting to English.")
            return "english"

def generate_caption(image):
    """Generates an English caption using the BLIP model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def capture_text_image():
    """Captures an image from the camera for OCR."""
    cap = initialize_camera()
    if cap is None:
        return None
    
    cv2.namedWindow('Capture Text', cv2.WINDOW_NORMAL)
    print("Position the textbook in front of the camera and press 's' to capture the image.")
    image_path = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Capture Text', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_path = "captured_text.jpg"
            cv2.imwrite(image_path, frame)
            print("Image captured.")
            break
        elif key == ord('q'):
            print("Quitting without capturing image.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return image_path

def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(gray)
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text

def main():
    cap = initialize_camera()
    if cap is None:
        return
    
    cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        cv2.imshow('Live Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting.")
            break
        elif key == ord('c'):
            print("Capturing image for OCR...")
            image_path = capture_text_image()
            if image_path:
                extracted_text = extract_text(image_path)
                print("Extracted Text:", extracted_text)
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
