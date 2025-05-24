import pickle
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class MultimodalModel:
    def __init__(self):
        """Initialize the LangChain-based multimodal model."""
        # Check for API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        # Initialize LangChain chat model
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )
        # System prompt for the assistant
        self.system_prompt = """
        You are a helpful assistant that can understand both text and images.
        When presented with images, describe what you see in detail.
        When answering questions, be concise but informative.
        If you don't know the answer, admit it rather than making something up.
        """
    
    def process_query(self, query_text, image=None):
        """Process a text query and optional image using LangChain."""
        # Create messages list starting with system prompt
        messages = [
            SystemMessage(content=self.system_prompt)
        ]
        
        # If image is provided, include it in the human message
        if image:
            # Create multimodal human message with text and image
            human_message_content = [
                {"type": "text", "text": query_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            ]
            messages.append(HumanMessage(content=human_message_content))
        else:
            # Text-only message
            messages.append(HumanMessage(content=query_text))
        
        # Get response from the model
        response = self.model.invoke(messages)
        
        return response.content


class TraditionalModel:
    def __init__(self, model_path="model/text_classifier.pkl", vectorizer_path="model/vectorizer.pkl"):
        """Initialize the traditional ML model with saved model and vectorizer."""
        self.model = self._load_model(model_path)
        self.vectorizer = self._load_model(vectorizer_path)
        self.corpus_df = None
        
    def _load_model(self, path):
        """Load a saved model or vectorizer."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return None
    
    def load_corpus(self, corpus_path="preprocessed_corpus.csv"):
        """Load preprocessed corpus for reference."""
        try:
            self.corpus_df = pd.read_csv(corpus_path)
            print(f"Loaded corpus with {len(self.corpus_df)} entries and {len(self.corpus_df['label'].unique())} categories")
            return True
        except Exception as e:
            print(f"Error loading corpus: {e}")
            return False
    
    def _preprocess_text(self, text):
        """Simple preprocessing for query text."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def process_query(self, query_text, image=None):
        """Process a text query and optional image."""
        if not self.model or not self.vectorizer:
            return "Model not properly loaded."
        
        # Preprocess the query text
        processed_text = self._preprocess_text(query_text)
        
        # Vectorize the processed text
        query_vector = self.vectorizer.transform([processed_text])
        
        # Get prediction
        prediction = self.model.predict(query_vector)[0]
        
        # Get confidence score
        confidence_scores = self.model.predict_proba(query_vector)[0]
        max_confidence = max(confidence_scores) * 100
        max_confidence_index = confidence_scores.argmax()
        
        # Get label for the highest confidence
        labels = self.model.classes_
        predicted_label = labels[max_confidence_index]
        
        # Image analysis message (placeholder)
        image_info = ""
        if image:
            image_info = "\n\nImage analysis: I've processed your image, but detailed image analysis is limited with this model."
        
        # Get relevant examples from corpus
        relevant_examples = self._get_relevant_examples(prediction)
        
        # Format response
        response = self._format_response(query_text, prediction, max_confidence, relevant_examples, image_info)
        
        return response
    
    def _get_relevant_examples(self, category, num_examples=2):
        """Get relevant examples from the corpus for the predicted category."""
        if self.corpus_df is None:
            return []
        
        # Filter corpus by predicted category
        category_df = self.corpus_df[self.corpus_df['label'] == category]
        
        if len(category_df) == 0:
            return []
        
        # Get random examples
        if len(category_df) <= num_examples:
            examples = category_df['text'].tolist()
        else:
            examples = category_df.sample(num_examples)['text'].tolist()
        
        return examples
    
    def _format_response(self, query, prediction, confidence, examples, image_info):
        """Format the model's response."""
        response = f"Based on your query: '{query}'\n\n"
        response += f"I've classified this as: {prediction}\n"
        response += f"Confidence: {confidence:.2f}%\n"
        
        if examples:
            response += "\nSimilar examples from our corpus:\n"
            for i, example in enumerate(examples, 1):
                response += f"{i}. {example}\n"
        
        if image_info:
            response += image_info
        
        return response