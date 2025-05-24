import cv2
import base64
import os
import json
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from threading import Lock, Thread
from playsound import playsound
from gtts import gTTS
import speech_recognition as sr
from wordcloud import WordCloud
from dotenv import load_dotenv
from model import MultimodalModel, TraditionalModel
from corpus_preprocessor import CorpusPreprocessor

class WebcamStream:
    def __init__(self):
        """Initialize webcam stream for capturing video."""
        self.stream = cv2.VideoCapture(0)  # Open webcam
        if not self.stream.isOpened():  # Check if the webcam is accessible
            print("❌ Error: Could not open webcam.")
            self.stream = None  # Set to None if it fails
            return
        
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()
        self.start_time = None
        self.frame_count = 0
        self.metrics = {
            "total_frames": 0,
            "fps_history": [],
            "response_times": []
        }

    def start(self):
        """Start the webcam stream in a separate thread."""
        if self.running or self.stream is None:
            return None
        self.running = True
        self.start_time = time.time()
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self  

    def read(self, encode=False):
        """Read the current frame from the webcam."""
        if self.stream is None:
            return None
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer).decode('utf-8')
        return frame

    def update(self):
        """Update the webcam frame continuously."""
        while self.running:
            _, self.frame = self.stream.read()
            self.frame_count += 1
            self.metrics["total_frames"] += 1
            
            # Calculate FPS every second
            if self.start_time and time.time() - self.start_time >= 1:
                fps = self.frame_count / (time.time() - self.start_time)
                self.metrics["fps_history"].append(fps)
                self.frame_count = 0
                self.start_time = time.time()

    def stop(self):
        """Stop the webcam stream."""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Clean up resources on exit."""
        if self.stream:
            self.stream.release()

    def get_metrics(self):
        """Get performance metrics for the webcam stream."""
        return self.metrics


class MultimodalSystem:
    def __init__(self):
        """Initialize the multimodal system."""
        # Load environment variables
        load_dotenv()
        
        self.model = None
        self.model_type = None  # 'langchain' or 'traditional'
        self.webcam_stream = None
        self.system_metrics = {
            "queries_processed": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "run_count": 0
        }
        self.corpus_df = None
        self.corpus_preprocessor = None
    
    def initialize_langchain_model(self):
        """Initialize the LangChain model."""
        try:
            self.model = MultimodalModel()
            self.model_type = 'langchain'
            print("✅ LangChain model initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing LangChain model: {e}")
            return False
    
    def initialize_traditional_model(self, model_path="model/text_classifier.pkl", vectorizer_path="model/vectorizer.pkl"):
        """Initialize the traditional ML model from saved files."""
        try:
            self.model = TraditionalModel(model_path, vectorizer_path)
            self.model_type = 'traditional'
            success = self.model.load_corpus()
            if success:
                print("✅ Traditional model initialized successfully!")
                return True
            else:
                print("❌ Failed to load corpus for the traditional model.")
                return False
        except Exception as e:
            print(f"❌ Error initializing traditional model: {e}")
            return False
    
    def preprocess_corpus(self, corpus_path):
        """Preprocess the corpus and train a traditional model."""
        try:
            self.corpus_preprocessor = CorpusPreprocessor()
            print("Beginning corpus preprocessing and model training...")
            
            # Preprocess corpus and train model
            model, vectorizer, self.corpus_df = self.corpus_preprocessor.process_and_train(corpus_path)
            
            if model and vectorizer:
                print("✅ Model successfully trained and saved.")
                
                # Visualize corpus data
                self._visualize_corpus_data()
                
                # Initialize the trained model
                success = self.initialize_traditional_model()
                return success
            else:
                print("❌ Failed to train model from corpus.")
                return False
                
        except Exception as e:
            print(f"❌ Error in corpus preprocessing: {e}")
            return False
    
    def _visualize_corpus_data(self):
        """Create visualizations of the corpus data."""
        if self.corpus_df is None or self.corpus_df.empty:
            print("No corpus data available for visualization.")
            return
        
        print("\nGenerating corpus visualizations...")
        
        # Create a directory for visualizations if it doesn't exist
        viz_dir = "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Label distribution
        plt.figure(figsize=(10, 6))
        label_counts = self.corpus_df['label'].value_counts()
        sns.barplot(x=label_counts.index, y=label_counts.values)
        plt.title('Distribution of Labels in Corpus')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'label_distribution.png'))
        plt.close()
        
        # 2. Word cloud
        all_text = ' '.join(self.corpus_df['text'])
        plt.figure(figsize=(10, 10))
        wordcloud = WordCloud(width=800, height=800, background_color='white', 
                              min_font_size=10).generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'wordcloud.png'))
        plt.close()
        
        # 3. Text length distribution
        plt.figure(figsize=(10, 6))
        text_lengths = self.corpus_df['text'].str.len()
        sns.histplot(text_lengths, bins=30)
        plt.title('Distribution of Text Lengths in Corpus')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'text_length_distribution.png'))
        plt.close()
        
        print(f"✅ Visualizations saved in '{viz_dir}' directory")
    
    def _tts(self, response):
        """Convert text response to speech using Google TTS."""
        try:
            tts = gTTS(response, lang="en")  
            temp_audio_path = os.path.join(os.getcwd(), "response.mp3")  
            tts.save(temp_audio_path) 

            playsound(temp_audio_path)
            os.remove(temp_audio_path)
            return True
        except Exception as e:
            print(f"❌ Error in text-to-speech: {e}")
            return False
    
    def process_query(self, query, image=None):
        """Process a query with the model and convert response to speech."""
        if not self.model:
            print("❌ Model not initialized. Please initialize a model first.")
            return None, 0
        
        # Record start time for metrics
        start_time = time.time()
        
        # Process query with the model
        response = self.model.process_query(query, image)
        
        # Update metrics
        response_time = time.time() - start_time
        self.system_metrics["queries_processed"] += 1
        self.system_metrics["total_response_time"] += response_time
        self.system_metrics["avg_response_time"] = (
            self.system_metrics["total_response_time"] / self.system_metrics["queries_processed"]
        )
        
        if self.webcam_stream and hasattr(self.webcam_stream, "metrics"):
            self.webcam_stream.metrics["response_times"].append(response_time)
        
        print("\n🤖 Response:", response)
        
        # Convert response to speech
        self._tts(response)
        
        return response, response_time
    
    def _audio_callback(self, recognizer, audio):
        """Process audio input when detected."""
        print("\n🎤 Audio detected, processing...")

        try:
            # Recognize speech using Google's speech recognition
            prompt = recognizer.recognize_google(audio)
            print(f"🎤 Recognized Text: '{prompt}'")

            if prompt:
                # Capture image if webcam available
                image_data = None
                if self.webcam_stream:
                    image_data = self.webcam_stream.read(encode=True)
                    if image_data:
                        print("📸 Image captured from webcam.")
                
                # Process query
                self.process_query(prompt, image_data)
            else:
                print("❌ No speech detected.")

        except sr.UnknownValueError:
            print("❌ Error: Could not understand the audio.")
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
    
    def run_interactive_system(self):
        """Run the interactive multimodal system."""
        if not self.model:
            print("❌ Model not initialized. Please initialize a model first.")
            return False
        
        # Initialize webcam
        self.webcam_stream = WebcamStream().start()
        if not self.webcam_stream or not self.webcam_stream.stream:
            print("⚠️ Warning: Could not initialize webcam. Continuing without visual input.")
        else:
            print("✅ Webcam initialized successfully.")
        
        # Speech recognition setup
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        try:
            with microphone as source:
                print("Adjusting for ambient noise. Please wait...")
                recognizer.adjust_for_ambient_noise(source)
                print("✅ Microphone initialized successfully.")

            # Start listening in background
            stop_listening = recognizer.listen_in_background(
                microphone, 
                lambda recognizer, audio: self._audio_callback(recognizer, audio)
            )
            
            print("\n✨ Running Multimodal System ✨")
            print(f"📊 Current model: {self.model_type.upper()}")
            print("🗣️  Speak to interact with the system")
            print("⌨️  Press 'q' or 'ESC' to exit\n")

            running = True
            while running:
                if self.webcam_stream:
                    frame = self.webcam_stream.read()
                    if frame is not None:
                        # Add instruction text to the frame
                        cv2.putText(frame, f"Model: {self.model_type.upper()}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        cv2.putText(frame, "Speak to interact with the system", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "Press 'q' or ESC to exit", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        cv2.imshow("Multimodal AI System", frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord("q")]:  # Exit on ESC or 'q'
                    running = False
                    print("Exiting system...")

            # Clean up
            if self.webcam_stream:
                self.webcam_stream.stop()
            cv2.destroyAllWindows()
            stop_listening(wait_for_stop=False)
            
            # Update run count
            self.system_metrics["run_count"] += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Error running interactive system: {e}")
            return False
    
    def display_metrics(self):
        """Display system performance metrics."""
        print("\n=== Performance Metrics ===")
        
        if self.system_metrics["run_count"] == 0:
            print("No metrics available. Run the system first.")
            return
        
        print(f"\nSystem has been run {self.system_metrics['run_count']} times")
        print(f"Total queries processed: {self.system_metrics['queries_processed']}")
        print(f"Average response time: {self.system_metrics['avg_response_time']:.2f} seconds")
        print(f"Model type used: {self.model_type}")
        
        if self.webcam_stream and hasattr(self.webcam_stream, "metrics"):
            webcam_metrics = self.webcam_stream.get_metrics()
            print(f"\nWebcam Metrics:")
            print(f"Total frames processed: {webcam_metrics['total_frames']}")
            
            if webcam_metrics["fps_history"]:
                avg_fps = sum(webcam_metrics["fps_history"]) / len(webcam_metrics["fps_history"])
                print(f"Average FPS: {avg_fps:.2f}")
        
        # Visualization option
        if self.system_metrics["queries_processed"] > 0:
            visualize = input("\nDo you want to visualize the metrics? (y/n): ")
            if visualize.lower() == 'y':
                self._display_metrics_visualizations()
    
    def _display_metrics_visualizations(self):
        """Display visualizations of the performance metrics."""
        # Response time histogram if we have data
        if self.webcam_stream and hasattr(self.webcam_stream, "metrics") and self.webcam_stream.metrics["response_times"]:
            plt.figure(figsize=(10, 5))
            plt.hist(self.webcam_stream.metrics["response_times"], bins=10, color='teal')
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        # FPS over time if available
        if self.webcam_stream and hasattr(self.webcam_stream, "metrics") and len(self.webcam_stream.metrics["fps_history"]) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(self.webcam_stream.metrics["fps_history"], color='orange')
            plt.title('FPS over Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frames Per Second')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def run_main_menu(self):
        """Display and handle the main menu."""
        while True:
            print("\n" + "="*60)
            print("             MULTIMODAL AI SYSTEM - MAIN MENU")
            print("="*60)
            print("1. Initialize LangChain Model")
            print("2. Load & Preprocess Corpus + Train Traditional Model")
            print("3. Load Existing Traditional Model")
            print("4. Run Interactive System")
            print("5. Show Performance Metrics")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == "1":
                # Initialize LangChain model
                success = self.initialize_langchain_model()
                if success:
                    print("✅ LangChain model initialized successfully.")
                else:
                    print("❌ Failed to initialize LangChain model. Please check your API key.")
                
            elif choice == "2":
                # Process corpus and train traditional model
                corpus_path = input("Enter the path to your JSON corpus file: ")
                if not os.path.exists(corpus_path):
                    print(f"❌ Error: File '{corpus_path}' not found.")
                    continue
                    
                success = self.preprocess_corpus(corpus_path)
                if success:
                    print("✅ Corpus processed and traditional model trained successfully.")
                else:
                    print("❌ Failed to process corpus or train model.")
                
            elif choice == "3":
                # Load existing traditional model
                model_path = input("Enter path to model file (default: model/text_classifier.pkl): ") or "model/text_classifier.pkl"
                vectorizer_path = input("Enter path to vectorizer file (default: model/vectorizer.pkl): ") or "model/vectorizer.pkl"
                
                if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                    print("❌ Error: Model or vectorizer file not found.")
                    continue
                    
                success = self.initialize_traditional_model(model_path, vectorizer_path)
                if success:
                    print("✅ Traditional model loaded successfully.")
                else:
                    print("❌ Failed to load traditional model.")
                
            elif choice == "4":
                # Run interactive system
                if not self.model:
                    print("❌ Model not initialized. Please choose option 1, 2, or 3 first.")
                    continue
                    
                self.run_interactive_system()
                
            elif choice == "5":
                # Show performance metrics
                self.display_metrics()
                
            elif choice == "6":
                # Exit
                print("\nExiting program. Goodbye!")
                break
                
            else:
                print("\n❌ Invalid choice. Please enter a number between 1 and 6.")

# Main entry point
if __name__ == "__main__":
    print("🚀 Starting Multimodal AI System...")
    system = MultimodalSystem()
    system.run_main_menu()











