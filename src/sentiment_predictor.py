from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Any

class SentimentPredictor:
    """
    A class to predict sentiment for both product and video comments
    using pre-trained BERT models.
    """
    def __init__(self, model_product_path: str, model_video_path: str, model_name: str = "bert-base-uncased"):
        """
        Initializes the SentimentPredictor by loading the tokenizer and two sentiment models.

        Args:
            model_product_path (str): Path to the directory containing the fine-tuned
                                      model for product sentiment (e.g., "./models/sentiment_for_product/").
            model_video_path (str): Path to the directory containing the fine-tuned
                                    model for video sentiment (e.g., "./models/sentiment_for_video/").
            model_name (str): The base pre-trained model name for the tokenizer
                              (default: "bert-base-uncased").
        """
        self.labels = ["negative", "neutral", "positive"]

        # Load the tokenizer once for both models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load both fine-tuned models
        self.model_product = AutoModelForSequenceClassification.from_pretrained(model_product_path)
        self.model_video = AutoModelForSequenceClassification.from_pretrained(model_video_path)

        # Set models to evaluation mode
        self.model_product.eval()
        self.model_video.eval()

        # Check for CUDA availability and move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_product.to(self.device)
        self.model_video.to(self.device)
        print(f"Models loaded and moved to {self.device} device.")

    def predict(self, comment_text: str) -> Dict[str, str]:
        """
        Predicts the sentiment for a given comment text for both product and video.

        Args:
            comment_text (str): The comment text. 

        Returns:
            Dict[str, str]: A dictionary containing predicted sentiments:
                            {"sentiment_for_product": "sentiment_label",
                             "sentiment_for_video": "sentiment_label"}
        """
        if not isinstance(comment_text, str):
            raise TypeError("Input 'comment_text' must be a string.")
        if not comment_text.strip():
            return {
                "sentiment_for_product": "neutral", # Or appropriate default/error handling
                "sentiment_for_video": "neutral"
            }

        # Tokenize input and move to the appropriate device
        inputs = self.tokenizer(comment_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Predict for product sentiment
        with torch.no_grad():
            outputs_product = self.model_product(**inputs)
            # Apply softmax to get probabilities if needed for confidence scores, otherwise argmax on logits is fine
            # probabilities_product = torch.softmax(outputs_product.logits, dim=1)
            pred_product_id = torch.argmax(outputs_product.logits, dim=1).item()
            sentiment_product = self.labels[pred_product_id]

        # Predict for video sentiment
        with torch.no_grad():
            outputs_video = self.model_video(**inputs)
            # probabilities_video = torch.softmax(outputs_video.logits, dim=1)
            pred_video_id = torch.argmax(outputs_video.logits, dim=1).item()
            sentiment_video = self.labels[pred_video_id]

        return {
            "sentiment_for_product": sentiment_product,
            "sentiment_for_video": sentiment_video
        }


    def clean_text(self, text, do_stemming=False): 
        # Convert all text to lowercase
        text = text.lower()

        # eplace all characters except letters, digits, and '?''!', with a blank space.
        text = re.sub('[^a-z0-9?!]', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()

        if self.do_stemming: 
            parts = text.split()
            parts = [self.ps.stem(part) for part in parts]
            text = ' '.join(parts)
        
        return text

# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Replace these paths with the actual paths where your fine-tuned models are saved.
    # For instance, if you saved them in your /workspace/ folder:
    product_model_path = "./models/sentiment_for_product/" # Or wherever your product model is saved
    video_model_path = "./models/sentiment_for_video/"   # Or wherever your video model is saved

    try:
        predictor = SentimentPredictor(
            model_product_path=product_model_path,
            model_video_path=video_model_path
        )

        # Test comments
        comment1 = "This product is amazing, completely changed my life!"
        comment2 = "The video was so boring and unhelpful, what a waste of time."
        comment3 = "The product is okay, but the video was pretty good."
        comment4 = "This comment is irrelevant to both product and video."
        comment5 = "" # Empty comment test

        print(f"Comment: '{comment1}'")
        print(f"Prediction: {predictor.predict(comment1)}\n")

        print(f"Comment: '{comment2}'")
        print(f"Prediction: {predictor.predict(comment2)}\n")

        print(f"Comment: '{comment3}'")
        print(f"Prediction: {predictor.predict(comment3)}\n")

        print(f"Comment: '{comment4}'")
        print(f"Prediction: {predictor.predict(comment4)}\n")
        
        print(f"Comment: '{comment5}'")
        print(f"Prediction: {predictor.predict(comment5)}\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your model paths are correct and PyTorch/Transformers are installed.")
        print("Example paths: './models/sentiment_for_product/' if 'models' is in your current directory.")