import tensorflow as tf
import numpy as np
from PIL import Image
from model_utils import preprocess_image, map_imagenet_to_animals
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimalClassifier:
    """
    Animal classification using pre-trained MobileNetV2 model
    """
    
    def __init__(self):
        """Initialize the classifier with pre-trained model"""
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained MobileNetV2 model"""
        try:
            logger.info("Loading MobileNetV2 model...")
            
            # Load pre-trained MobileNetV2 model with ImageNet weights
            self.model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")
    
    def predict(self, image, debug_mode=False):
        """
        Predict the animal in the given image
        
        Args:
            image: PIL Image object
            debug_mode: If True, includes raw ImageNet predictions for debugging
        
        Returns:
            tuple: (predicted_animal, confidence_percentage, top_predictions_list)
        """
        try:
            if self.model is None:
                raise Exception("Model not loaded")
            
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # If debug mode, get raw ImageNet predictions
            raw_predictions = []
            if debug_mode:
                # Get top 10 raw ImageNet predictions for debugging
                decoded = tf.keras.applications.imagenet_utils.decode_predictions(
                    predictions, top=10
                )[0]
                raw_predictions = [(pred[1].replace('_', ' ').title(), pred[2] * 100) for pred in decoded]
            
            # Map predictions to animal names
            animal_predictions = map_imagenet_to_animals(predictions, top_k=5)
            
            if not animal_predictions:
                # If no animals detected, try to get general predictions
                top_idx = np.argmax(predictions[0])
                confidence = predictions[0][top_idx] * 100
                
                # Decode ImageNet predictions to get class names
                decoded = tf.keras.applications.imagenet_utils.decode_predictions(
                    predictions, top=1
                )[0]
                
                if decoded:
                    class_name = decoded[0][1].replace('_', ' ').title()
                    if debug_mode:
                        raw_predictions = [(pred[1].replace('_', ' ').title(), pred[2] * 100) for pred in decoded[:5]]
                        return class_name, confidence, [(class_name, confidence)], raw_predictions
                    else:
                        return class_name, confidence, [(class_name, confidence)]
                else:
                    if debug_mode:
                        return None, 0, [], []
                    else:
                        return None, 0, []
            
            # Return the top prediction and all top predictions
            top_animal, top_confidence = animal_predictions[0]
            
            if debug_mode:
                return top_animal, top_confidence, animal_predictions, raw_predictions
            else:
                return top_animal, top_confidence, animal_predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None, 0, []
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "model_name": "MobileNetV2",
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "total_params": self.model.count_params(),
            "status": "Loaded and ready"
        }
    
    def is_ready(self):
        """
        Check if the classifier is ready for predictions
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.model is not None
