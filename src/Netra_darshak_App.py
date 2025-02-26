from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import pytesseract
from gtts import gTTS
import os
import threading
from plyer import tts

class NetraDarshakApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)
        
        self.label = Label(text="Press Capture to Scan")
        self.layout.add_widget(self.label)
        
        self.button = Button(text="Capture & Read")
        self.button.bind(on_press=self.capture_image)
        self.layout.add_widget(self.button)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout
    
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
    
    def capture_image(self, instance):
        ret, frame = self.capture.read()
        if ret:
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            extracted_text = self.extract_text(image_path)
            self.label.text = extracted_text
            threading.Thread(target=self.speak_text, args=(extracted_text,)).start()
    
    def extract_text(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text if text.strip() else "No text detected"
    
    def speak_text(self, text):
        tts.speak(text)
    
    def on_stop(self):
        self.capture.release()

if __name__ == "__main__":
    NetraDarshakApp().run()
