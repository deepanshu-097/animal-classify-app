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
            
            # Map predictions to animal names (specialized for cow/buffalo)
            animal_predictions = map_imagenet_to_animals(predictions, top_k=3)
            
            if not animal_predictions:
                # Enhanced fallback for cow/buffalo detection
                # Look for bovine-related terms in raw predictions
                decoded = tf.keras.applications.imagenet_utils.decode_predictions(
                    predictions, top=20  # Check more predictions for bovine terms
                )[0]
                
                bovine_keywords = [
                    'ox', 'bull', 'cow', 'cattle', 'buffalo', 'bison', 'zebu', 'water_buffalo',
                    'bovine', 'steer', 'heifer', 'calf', 'dairy', 'beef', 'holstein', 'jersey',
                    'angus', 'brahman', 'hereford', 'longhorn', 'shorthorn', 'highland',
                    'yak', 'gaur', 'banteng', 'gayal', 'kouprey', 'aurochs',
                    'cape_buffalo', 'african_buffalo', 'water_ox', 'swamp_buffalo',
                    'carabao', 'murrah', 'nili_ravi', 'surti'
                ]
                
                for pred in decoded:
                    class_name = pred[1].lower().replace('_', ' ')
                    confidence = pred[2] * 100
                    
                    # Check if any bovine keywords are in the class name
                    for keyword in bovine_keywords:
                        if keyword in class_name:
                            if 'buffalo' in class_name or 'bison' in class_name or 'water' in class_name:
                                result_animal = 'Buffalo'
                            else:
                                result_animal = 'Cow'
                            
                            # Boost confidence for clearly bovine detections
                            boosted_confidence = min(confidence * 2.5, 95.0)  # Cap at 95%
                            
                            # Additional boost if multiple bovine keywords found
                            keyword_count = sum(1 for kw in bovine_keywords if kw in class_name)
                            if keyword_count > 1:
                                boosted_confidence = min(boosted_confidence * 1.2, 95.0)
                            
                            if debug_mode:
                                raw_predictions = [(p[1].replace('_', ' ').title(), p[2] * 100) for p in decoded[:5]]
                                return result_animal, boosted_confidence, [(result_animal, boosted_confidence)], raw_predictions
                            else:
                                return result_animal, boosted_confidence, [(result_animal, boosted_confidence)]
                
                # If no bovine terms found, reject the image
                if debug_mode:
                    raw_predictions = [(pred[1].replace('_', ' ').title(), pred[2] * 100) for pred in decoded[:5]]
                    return "Not a cow or buffalo", 0, [("Not a cow or buffalo", 0)], raw_predictions
                else:
                    return "Not a cow or buffalo", 0, [("Not a cow or buffalo", 0)]
            
            # Enhanced accuracy logic for cow/buffalo classification
            if animal_predictions:
                top_animal, top_confidence = animal_predictions[0]
                
                # Apply stricter confidence thresholds for 99% accuracy goal
                if top_confidence >= 85:  # High confidence threshold
                    confidence_level = "High"
                elif top_confidence >= 70:  # Medium confidence threshold  
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                    top_animal = f"Uncertain - possibly {top_animal}"
                
                if debug_mode:
                    return top_animal, top_confidence, animal_predictions, raw_predictions
                else:
                    return top_animal, top_confidence, animal_predictions
            else:
                if debug_mode:
                    return "Not a cow or buffalo", 0, [], raw_predictions
                else:
                    return "Not a cow or buffalo", 0, []
            
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
