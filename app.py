import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the ensemble model
model = load_model('ensemble_model.h5')

# Mapping predicted classes to rice types
rice_types = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}

# Preprocessing function
def preprocess_image(image):
    img = image.resize((28, 28))  # Resize to model's input shape
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img_array

# Add a custom background and title
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        color: #4B6587;
    }
    .main {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4B6587;
        color: white;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #2C3E50;">RiceVision</h1>
    <h3 style="color: #34495E;">AI-Powered Rice Grain Classifier</h3>
    <p style="color: #7D8891;">
        Upload an image of rice grains, and our advanced AI model will classify it into one of five types: Arborio, Basmati, Ipsala, Jasmine, or Karacadag.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About RiceVision")
st.sidebar.info("""
RiceVision uses a cutting-edge ensemble model combining CNN and LSTM architectures to classify rice types. 
Supported types:
- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag
""")

# File Uploader
st.markdown("<h4 style='color: #34495E;'>Upload your rice images below:</h4>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Choose images (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Results and Visualization
if uploaded_files:
    results = []

    # Process uploaded files
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f"Uploaded Image: {file.name}", use_container_width=True)

        try:
            # Preprocess and Predict
            img_array, additional_input = preprocess_image(image)
            predictions = model.predict([img_array, additional_input])
            predicted_class = np.argmax(predictions, axis=1)[0]
            rice_name = rice_types.get(predicted_class, "Unknown")
            confidence_scores = predictions[0] * 100

            # Store result
            results.append({'Filename': file.name, 'Predicted Rice Type': rice_name})

            # Display Prediction
            st.markdown(f"<h4 style='color: #2C3E50;'>Prediction: {rice_name}</h4>", unsafe_allow_html=True)
            st.markdown("<b>Confidence Scores:</b>", unsafe_allow_html=True)

            # Bar Chart for Confidence Levels
            prob_df = pd.DataFrame({
                'Rice Type': list(rice_types.values()),
                'Confidence (%)': confidence_scores
            })
            fig = px.bar(
                prob_df,
                x='Rice Type',
                y='Confidence (%)',
                color='Rice Type',
                title="Confidence Distribution",
                text_auto='.2f'
            )
            fig.update_layout(
                xaxis_title="Rice Types",
                yaxis_title="Confidence (%)",
                showlegend=False
            )
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # Results Table
    if results:
        st.write("### Results Summary")
        df = pd.DataFrame(results)
        st.write(df)

        # CSV Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
            help="Click to download the prediction results."
        )

# Optional: Confusion Matrix
if st.sidebar.checkbox("Show Confusion Matrix (Requires labeled test data)"):
    # Example test data (replace with actual test data)
    y_true = [0, 1, 2, 3, 4, 0]  # Replace with true labels
    y_pred = [0, 1, 1, 3, 4, 0]  # Replace with model predictions

    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(conf_matrix, display_labels=list(rice_types.values())).plot(ax=ax, cmap='Blues')
    st.pyplot(fig)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 20px; color: #95A5A6;">
    <small>© 2024 RiceVision. Developed by Pradeep Kumar R.</small>
</div>
""", unsafe_allow_html=True)
