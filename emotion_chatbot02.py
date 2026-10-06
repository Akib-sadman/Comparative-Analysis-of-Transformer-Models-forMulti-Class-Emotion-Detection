"""
Emotion-Aware Chatbot v2 - Using XLNet on Empathetic Dataset
Predicts emotions from 32 emotion categories and generates conversational responses with Gemini API.
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai


# Check if Gemini is available
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: google-genai not installed. Gemini integration disabled.")
    print("Install with: pip install google-genai")

# Paths - Using trained XLNet model on Empathetic dataset
TRAINED_MODEL_PATH = "./results/xlnet_empathetic/checkpoint-9690"  # Best checkpoint
MAX_LENGTH = 128

# Emotion labels (Empathetic Dialogue dataset - 32 emotions)
# Cleaned from empathetic_label_map.csv (rows with proper emotion names)
EMOTION_LABELS = {
    6: "afraid",
    7: "angry",
    8: "annoyed",
    9: "anticipating",
    10: "anxious",
    11: "apprehensive",
    12: "ashamed",
    13: "caring",
    14: "confident",
    15: "content",
    16: "devastated",
    17: "disappointed",
    18: "disgusted",
    19: "embarrassed",
    20: "excited",
    21: "faithful",
    22: "furious",
    23: "grateful",
    24: "guilty",
    25: "hopeful",
    26: "impressed",
    27: "jealous",
    28: "joyful",
    29: "lonely",
    31: "nostalgic",
    32: "prepared",
    33: "proud",
    34: "sad",
    35: "sentimental",
    36: "surprised",
    41: "terrified",
    42: "trusting"
}


class EmotionChatbot:
    def __init__(self):
        """Initialize the chatbot with trained XLNet emotion prediction model (Empathetic)."""
        print("=" * 70)
        print("🤖 EMOTION-AWARE CHATBOT v2 (EMPATHETIC DATASET)")
        print("=" * 70)
        print("Initializing...")
        
        # Check device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load trained XLNet model
        print(f"Loading trained XLNet model from: {TRAINED_MODEL_PATH}")
        
        if not os.path.exists(TRAINED_MODEL_PATH):
            print(f"\n❌ ERROR: Model not found at {TRAINED_MODEL_PATH}")
            print("Please ensure the trained model exists.")
            sys.exit(1)
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                TRAINED_MODEL_PATH,
                num_labels=43,  # Full label count in model
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("✓ Model loaded successfully!")
            print("  Model: XLNet (xlnet-base-cased)")
            print("  Dataset: Empathetic Dialogue")
            print("  Emotions: 32 categories")
            print("  Accuracy: 36.8% | F1: 36.2%")
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            sys.exit(1)
        
        # Load emotion labels
        self.emotion_labels = EMOTION_LABELS
        
        # Initialize Gemini
        self.setup_gemini()
        
        print("=" * 70)
        print("✓ Chatbot ready!")
        print("=" * 70)
    
    def setup_gemini(self):
        """Setup Gemini API with user's API key."""
        if not GEMINI_AVAILABLE:
            self.gemini_client = None
            return
        
        # Your API key (hardcoded for convenience)
        api_key = "AIzaSyA4yFLVBVkjkBteE5d70C02yMgZXOuFZmw"
        
        # Override with environment variable if set
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key:
            api_key = env_key
        
        # Use the new google.genai Client API
        try:
            # Use the new google.genai Client API
            self.gemini_client = genai.Client(api_key=api_key)
            
            # Test with a simple generation
            try:
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents='test'
                )
                print("\n✓ Gemini API configured successfully!\n")
            except Exception as test_error:
                print(f"\n⚠ Gemini API test failed: {test_error}")
                print("Continuing without Gemini integration.\n")
                self.gemini_client = None
                
        except Exception as e:
            print(f"\n⚠ Error configuring Gemini: {e}")
            print("Continuing without Gemini integration.\n")
            self.gemini_client = None
    
    def predict_emotion(self, text):
        """
        Predict emotion from text using trained XLNet model.
        
        Returns:
            predicted_emotion: str, the predicted emotion label
            confidence: float, confidence score (0-100)
            all_scores: dict, all emotion scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get scores for known emotions only
        probs = probabilities[0].cpu().numpy()
        all_scores = {}
        
        for idx, emotion_name in self.emotion_labels.items():
            all_scores[emotion_name] = float(probs[idx]) * 100
        
        # Get predicted emotion (from known emotions only)
        max_score = 0
        predicted_emotion = "unknown"
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        
        # Check if predicted index is in our known emotions
        if predicted_idx in self.emotion_labels:
            predicted_emotion = self.emotion_labels[predicted_idx]
            confidence = probs[predicted_idx] * 100
        else:
            # If not in known emotions, find the highest score among known emotions
            for idx, score in all_scores.items():
                if score > max_score:
                    max_score = score
                    predicted_emotion = idx
            confidence = max_score
        
        return predicted_emotion, confidence, all_scores
    
    def display_emotion_scores(self, emotion, confidence, all_scores):
        """Display emotion prediction with bar chart."""
        print("\n" + "🤖 " + "=" * 68)
        print(f"   Prediction: {emotion.upper()} (confidence: {confidence:.1f}%)")
        print("=" * 70)
        
        print("\n📊 All emotion scores:")
        
        # Sort scores by value for better visualization
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Show top 10 emotions (since 32 is too many to display nicely)
        print("   Top 10 emotions:")
        
        # Find max score for scaling the bars
        max_score = max(all_scores.values())
        bar_width = 50  # Width of the bar in characters
        
        for emotion_name, score in sorted_scores[:10]:
            # Calculate bar length
            bar_length = int((score / max_score) * bar_width) if max_score > 0 else 0
            bar = "█" * bar_length
            
            # Format emotion name (align left)
            emotion_display = f"{emotion_name:<18}"
            
            # Display
            print(f"   {emotion_display} {score:5.1f}% {bar}")
        
        print("")
    
    def get_gemini_response(self, user_text, emotion):
        """Get conversational response from Gemini based on text and emotion."""
        if not self.gemini_client:
            return None
        
        try:
            # Create prompt for Gemini
            prompt = f"""You are having a conversation with a user. Based on their message and detected emotion, provide a natural, empathetic conversational response.

User's message: "{user_text}"
Detected emotion: {emotion}

Respond naturally to continue the conversation. Keep it concise (2-3 sentences)."""
            
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"⚠ Error getting Gemini response: {e}")
            return None
    
    def run(self):
        """Main chatbot loop."""
        print("\nType your text and press Enter to predict emotion.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            # Get user input
            try:
                user_input = input("💬 Enter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Predict emotion
            emotion, confidence, all_scores = self.predict_emotion(user_input)
            
            # Display results
            self.display_emotion_scores(emotion, confidence, all_scores)
            
            # Get Gemini response
            if self.gemini_client:
                print("🤔 Generating response...\n")
                gemini_response = self.get_gemini_response(user_input, emotion)
                
                if gemini_response:
                    print("=" * 70)
                    print("💬 Gemini Response:")
                    print("=" * 70)
                    print(f"   {gemini_response}")
                    print("=" * 70)
            
            print("\n")


def main():
    """Main entry point."""
    print("\n")
    
    # Check if model exists
    if not os.path.exists(TRAINED_MODEL_PATH):
        print("=" * 70)
        print("❌ ERROR: Trained XLNet model not found!")
        print("=" * 70)
        print(f"\nExpected model at: {TRAINED_MODEL_PATH}")
        print("\nPlease ensure your trained model checkpoint exists.")
        print("=" * 70)
        sys.exit(1)
    
    # Initialize and run chatbot
    try:
        chatbot = EmotionChatbot()
        chatbot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
