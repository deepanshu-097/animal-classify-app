import streamlit as st
import numpy as np
from PIL import Image
import io
from animal_classifier import AnimalClassifier
import traceback

# Configure page
st.set_page_config(
    page_title="AI Animal Recognition",
    page_icon="🐾",
    layout="centered"
)

# Initialize the classifier
@st.cache_resource
def load_classifier():
    """Load and cache the animal classifier model"""
    try:
        classifier = AnimalClassifier()
        return classifier
    except Exception as e:
        st.error(f"Failed to load the AI model: {str(e)}")
        return None

def main():
    st.title("🐾 AI Animal Recognition")
    st.write("Upload a photo of an animal and let AI identify it for you!")
    
    # Load the classifier
    classifier = load_classifier()
    
    if classifier is None:
        st.error("Unable to load the AI model. Please try again later.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an animal photo...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        help="Supported formats: JPG, JPEG, PNG, BMP, GIF"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Your uploaded photo", use_column_width=True)
            
            with col2:
                st.subheader("AI Prediction")
                
                # Show loading spinner while processing
                with st.spinner("Analyzing the image..."):
                    # Get prediction from the classifier
                    prediction, confidence, top_predictions = classifier.predict(image)
                
                if prediction:
                    # Display main prediction
                    st.success(f"**Detected Animal: {prediction}**")
                    st.write(f"**Confidence: {confidence:.1f}%**")
                    
                    # Display top 3 predictions
                    st.subheader("Top 3 Predictions:")
                    for i, (animal, conf) in enumerate(top_predictions[:3], 1):
                        st.write(f"{i}. {animal}: {conf:.1f}%")
                    
                    # Add confidence interpretation
                    if confidence >= 80:
                        st.info("🎯 High confidence prediction!")
                    elif confidence >= 60:
                        st.warning("⚠️ Moderate confidence. The image might be unclear or contain multiple animals.")
                    else:
                        st.error("❓ Low confidence. Please try a clearer image with a single animal.")
                else:
                    st.error("Unable to classify the image. Please try another photo.")
        
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            st.write("Please try uploading a different image.")
    
    # Add information section
    with st.expander("ℹ️ About this AI"):
        st.write("""
        **How it works:**
        - This AI uses a pre-trained MobileNetV2 model fine-tuned for animal recognition
        - It can identify common animals including dogs, cats, birds, farm animals, and wildlife
        - The model analyzes image features and provides confidence scores for predictions
        
        **Tips for best results:**
        - Use clear, well-lit photos
        - Ensure the animal is the main subject of the image
        - Avoid images with multiple animals
        - Higher resolution images generally work better
        
        **Supported animals:**
        Dogs, Cats, Birds, Horses, Cows, Sheep, Pigs, Elephants, Lions, Tigers, Bears, and many more!
        """)
    
    # Add sample images section
    with st.expander("📸 Try with sample images"):
        st.write("Don't have an animal photo? Try these sample images:")
        
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            if st.button("🐕 Sample Dog"):
                st.info("Please upload your own animal photo to test the AI!")
        
        with sample_col2:
            if st.button("🐱 Sample Cat"):
                st.info("Please upload your own animal photo to test the AI!")
        
        with sample_col3:
            if st.button("🐦 Sample Bird"):
                st.info("Please upload your own animal photo to test the AI!")

if __name__ == "__main__":
    main()
