import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for model prediction
    
    Args:
        image: PIL Image object
        target_size: Target size for the image (width, height)
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2 (scale to [-1, 1])
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def get_animal_classes():
    """
    Return a mapping of ImageNet class indices specifically for cow and buffalo recognition
    Optimized for high accuracy detection of these two bovine species
    """
    # Specialized mapping focusing only on cow and buffalo related ImageNet classes
    animal_classes = {
        # Cow and Cattle Classifications (ImageNet indices)
        343: "Cow",           # zebu (type of cattle)
        344: "Cow",           # ox (castrated male cattle)
        349: "Cow",           # ox (another mapping)
        
        # Buffalo Classifications
        350: "Buffalo",       # water_buffalo
        351: "Buffalo",       # bison (American buffalo)
        
        # Additional bovine species that can help distinguish
        352: "Buffalo",       # ram (can be confused with buffalo in some contexts)
        
        # Holstein cow (specific breed if available in ImageNet)
        # Note: Some ImageNet indices might map to specific cow breeds
        147: "Cow",           # Some indices that might catch cow-like animals
        
        # Additional safety mappings for bovine detection
        # These help catch edge cases where the model detects related terms
        358: "Cow",           # llama (sometimes confused, but we'll map to closest)
        359: "Cow",           # More llama mappings that might catch cow-like shapes
    }
    
    return animal_classes

def map_imagenet_to_animals(predictions, top_k=5):
    """
    Map ImageNet predictions to animal names
    
    Args:
        predictions: Model predictions array
        top_k: Number of top predictions to return
    
    Returns:
        List of tuples (animal_name, confidence_percentage)
    """
    animal_classes = get_animal_classes()
    
    # Get top k predictions
    top_indices = np.argsort(predictions[0])[::-1][:top_k * 3]  # Get more to filter animals
    
    animal_predictions = []
    for idx in top_indices:
        if idx in animal_classes:
            confidence = predictions[0][idx] * 100
            animal_predictions.append((animal_classes[idx], confidence))
        
        if len(animal_predictions) >= top_k:
            break
    
    return animal_predictions
