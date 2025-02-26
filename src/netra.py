import cv2
import torch
import threading
import time
import os
import speech_recognition as sr
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from playsound import playsound
from googletrans import Translator


# import properties.json
import json
with open("properties.json", "r") as f:
    properties = json.load(f)
    
    
# ------------------------------------




# -------------------------
# Setup for Scene Description (BLIP)
# -------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()  # Set model to evaluation mode

translator = Translator()
SPEAK_INTERVAL = 3  # Speak every 3 seconds
selected_language = 'en'  # Default language

LANGUAGE_CODES = {
    "english": "en",
    "telugu": "te",
    "hindi": "hi",
    "tamil": "ta",
    "kannada": "kn",
    "marathi": "mr",
    "bengali": "bn",
    "gujarati": "gu",
    "malayalam": "ml",
    "urdu": "ur"
}



def get_Device_from_MAC():
    MAC  = properties.get("MAC")
    # run arp -a in cmd and get the ip address of the device with the MAC address
    ip_address = ""
    os.system("arp -a > arp.txt")
    with open("arp.txt", "r") as f:
        for line in f:
            if MAC in line:
                ip_address = line.split()[0]
                break
    os.remove("arp.txt")
    return ip_address
    
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
            return LANGUAGE_CODES.get(detected_language, "en")
        except sr.UnknownValueError:
            print("Could not understand audio, defaulting to English.")
            return "en"
        except sr.RequestError:
            print("Speech Recognition service error, defaulting to English.")
            return "en"

def generate_caption(image):
    """Generates an English caption using the BLIP model."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def translate_text(text, lang_code):
    """Translates text to the chosen language."""
    try:
        return translator.translate(text, dest=lang_code).text
    except Exception as e:
        print("Translation Error:", e)
        return text

def speak_text(text, lang_code):
    """Converts text to speech and plays the audio."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        print("TTS Error:", e)

class CaptioningThread(threading.Thread):
    """Thread to continuously capture frames, generate captions, and speak."""
    def __init__(self, capture, interval, lang_code):
        super().__init__()
        self.capture = capture
        self.interval = interval
        self.lang_code = lang_code
        self.current_caption = ""
        self.running = True

    def run(self):
        try:
            last_time = time.time()
            while self.running:
                if time.time() - last_time >= self.interval:
                    ret, frame = self.capture.read()
                    if not ret:
                        continue
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    caption = generate_caption(rgb_frame)
                    translated_caption = translate_text(caption, self.lang_code)
                    if translated_caption != self.current_caption:
                        self.current_caption = translated_caption
                        print("English:", caption)
                        print(f"Translated ({self.lang_code}):", translated_caption)
                        threading.Thread(target=speak_text, args=(translated_caption, self.lang_code)).start()
                    last_time = time.time()
        except Exception as e:
            print("Exception in CaptioningThread:", e)

    def stop(self):
        self.running = False

# -------------------------
# Setup for OCR-based Textbook Reading
# -------------------------
def capture_text_image():
    """Captures an image from the camera for OCR."""
    ip = get_Device_from_MAC()
    cap = cv2.VideoCapture(ip+"/video")
    if not cap.isOpened():
        print("Error: Could not access webcam.")
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

def run_textbook_reading(lang='en'):
    """Captures an image of textbook text, extracts text via OCR, translates it, and reads it aloud."""
    image_path = capture_text_image()
    if image_path is None:
        return
    extracted_text = extract_text(image_path)
    if not extracted_text.strip():
        print("No text detected in the image.")
        return
    print("Extracted Text:")
    print(extracted_text)
    
    # Translate the extracted text into the language of description.
    translated_text = translate_text(extracted_text, lang)
    print("Translated Extracted Text:")
    print(translated_text)
    speak_text(translated_text, lang)

# -------------------------
# Main Program with Auto-switching
# -------------------------
def run_scene_description():
    """Starts the real-time scene description using the webcam."""
    global selected_language
    selected_language = recognize_language()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
    caption_thread = CaptioningThread(cap, SPEAK_INTERVAL, selected_language)
    caption_thread.start()
    print(f"Real-time scene description started in {selected_language}.")
    print("Press 'p' to capture a picture for OCR reading. Press 'q' to exit scene description.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow('Live Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting scene description.")
            break

        # When 'p' is pressed, capture picture for OCR reading
        if key == ord('p'):
            print("Picture capture requested for OCR reading.")
            caption_thread.stop()  # Stop scene description temporarily
            caption_thread.join()
            cap.release()
            cv2.destroyAllWindows()
            run_textbook_reading(lang=selected_language)
            print("Returning to scene description mode...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not access webcam.")
                return
            cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
            caption_thread = CaptioningThread(cap, SPEAK_INTERVAL, selected_language)
            caption_thread.start()

    caption_thread.stop()
    caption_thread.join()
    cap.release()
    cv2.destroyAllWindows()

def main():
    run_scene_description()

if __name__ == "__main__":
    main()  