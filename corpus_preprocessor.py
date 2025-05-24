import json
import nltk
import pandas as pd
import numpy as np
import re
import pickle
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class CorpusPreprocessor:
    def __init__(self):
        # Download necessary NLTK data
        self._download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                print(f"✅ Downloaded NLTK resource: {resource}")
            except Exception as e:
                print(f"❌ Error downloading {resource}: {e}")
    
    def preprocess_text(self, text):
        """Preprocess a single text string."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        # Join back to string for vectorization
        return " ".join(tokens)
    
    def preprocess_corpus(self, filepath):
        """Load and preprocess the corpus data."""
        try:
            with open(filepath, "r") as f:
                corpus = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: File '{filepath}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in '{filepath}'.")
            return None

        print("🔄 Processing corpus data...")
        
        # Create lists to store processed data
        labels = []
        processed_descriptions = []
        processed_speech = []
        image_paths = []
        
        for item in corpus:
            # Process each item in corpus
            label = item.get("label", "unknown")
            
            # Process descriptions
            for desc in item.get("descriptions", []):
                processed_text = self.preprocess_text(desc)
                if processed_text:
                    processed_descriptions.append(processed_text)
                    labels.append(label)
                    image_paths.append(item.get("image", ""))
            
            # Process speech
            for speech in item.get("speech", []):
                processed_text = self.preprocess_text(speech)
                if processed_text:
                    processed_speech.append(processed_text)
                    labels.append(label)
                    image_paths.append(item.get("image", ""))
        
        # Combine descriptions and speech for training
        all_text = processed_descriptions + processed_speech
        
        print(f"✅ Corpus preprocessing complete:")
        print(f"  - {len(processed_descriptions)} descriptions processed")
        print(f"  - {len(processed_speech)} speech samples processed")
        print(f"  - {len(set(labels))} unique labels found")
        
        # Create a DataFrame for easier handling
        corpus_df = pd.DataFrame({
            'text': all_text,
            'label': labels,
            'image_path': image_paths
        })
        
        return corpus_df
    
    def visualize_corpus(self, corpus_df):
        """Create and display visualizations of the corpus data."""
        if corpus_df is None or corpus_df.empty:
            print("❌ No corpus data available for visualization.")
            return
        
        # Ensure directory exists
        viz_dir = "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        print("📊 Generating corpus visualizations...")
        
        # 1. Label distribution
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(y='label', data=corpus_df, order=corpus_df['label'].value_counts().index)
        plt.title('Distribution of Labels in Corpus', fontsize=15)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Label', fontsize=12)
        # Add count labels
        for p in ax.patches:
            width = p.get_width()
            plt.text(width + 1, p.get_y() + p.get_height()/2, f'{int(width)}', 
                    ha='left', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'label_distribution.png'))
        plt.close()
        
        # 2. Text length analysis
        plt.figure(figsize=(10, 6))
        corpus_df['text_length'] = corpus_df['text'].str.len()
        sns.histplot(data=corpus_df, x='text_length', bins=30, kde=True)
        plt.title('Distribution of Text Lengths', fontsize=15)
        plt.xlabel('Text Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'text_length_distribution.png'))
        plt.close()
        
        # 3. Word cloud for top categories
        top_categories = corpus_df['label'].value_counts().head(5).index.tolist()
        for category in top_categories:
            category_text = ' '.join(corpus_df[corpus_df['label'] == category]['text'])
            if category_text.strip():
                plt.figure(figsize=(10, 8))
                wordcloud = WordCloud(width=800, height=400, 
                                      background_color='white', 
                                      max_words=100, 
                                      contour_width=3).generate(category_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Word Cloud for Category: {category}', fontsize=15)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'wordcloud_{category}.png'))
                plt.close()
        
        # 4. Average text length by category
        plt.figure(figsize=(12, 6))
        avg_lengths = corpus_df.groupby('label')['text_length'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_lengths.values, y=avg_lengths.index)
        plt.title('Average Text Length by Category', fontsize=15)
        plt.xlabel('Average Length (characters)', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'avg_text_length_by_category.png'))
        plt.close()
        
        print(f"✅ Visualizations saved in '{viz_dir}' directory")
    
    def train_model(self, corpus_df):
        """Train a classifier on the preprocessed corpus data."""
        if corpus_df is None or corpus_df.empty:
            print("❌ No corpus data available for training.")
            return None, None
        
        print("🔄 Training model on corpus data...")
        
        # Create directories for model storage
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)
        
        # Split data into features and target
        X = corpus_df['text']
        y = corpus_df['label']
        
        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        
        print(f"Training set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples")
        
        # Vectorize the text data
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.85
        )
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        
        # Train a Random Forest classifier
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        classifier.fit(X_train_vectorized, y_train)
        
        # Evaluate the model
        y_pred = classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model training complete. Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        model_path = os.path.join(model_dir, "text_classifier.pkl")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Vectorizer saved to {vectorizer_path}")
        
        # Save preprocessed corpus for reference
        corpus_df.to_csv("preprocessed_corpus.csv", index=False)
        print("✅ Preprocessed corpus saved to preprocessed_corpus.csv")
        
        return classifier, vectorizer
    
    def process_and_train(self, corpus_path):
        """Process corpus data and train a model on it."""
        # Preprocess corpus
        corpus_df = self.preprocess_corpus(corpus_path)
        if corpus_df is None:
            return None, None, None
        
        # Create visualizations
        self.visualize_corpus(corpus_df)
        
        # Train model
        model, vectorizer = self.train_model(corpus_df)
        
        return model, vectorizer, corpus_df