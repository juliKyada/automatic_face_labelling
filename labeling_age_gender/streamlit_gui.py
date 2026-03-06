import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import zipfile
import io
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import glob
from pathlib import Path
from typing import Optional, Tuple
from config import MODEL_PATH, ENABLE_FACE_DETECTION, FACE_DETECTION_SCALE_FACTOR, FACE_DETECTION_MIN_NEIGHBORS, FACE_DETECTION_MIN_SIZE

# Page configuration
st.set_page_config(
    page_title="🤖 Automatic Facial Image Labelling System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    .folder-upload {
        background-color: #e8f5e8;
        border-color: #28a745;
    }
    .file-upload {
        background-color: #e8f4fd;
        border-color: #1f77b4;
    }
    .stButton > button {
        border-radius: 20px;
        font-weight: bold;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    .folder-upload {
        background-color: #e8f5e8;
        border-color: #28a745;
    }
    .file-upload {
        background-color: #e8f4fd;
        border-color: #1f77b4;
    }
    .stButton > button {
        border-radius: 20px;
        font-weight: bold;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def upload_folder():
    """Create a folder upload interface"""
    st.markdown("### 📁 Upload Folder")
    
    # Option 1: Zip file upload
    st.markdown("**Option 1: Upload as ZIP file**")
    zip_file = st.file_uploader(
        "Upload a ZIP file containing your images",
        type=['zip'],
        help="Create a ZIP file with your images and upload it here"
    )
    
    if zip_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find all image files
                image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(temp_dir, '**', ext), recursive=True))
                
                if image_files:
                    st.success(f"✅ Found {len(image_files)} images in ZIP file")
                    return image_files
                else:
                    st.warning("⚠️ No image files found in ZIP file")
                    return []
        except Exception as e:
            st.error(f"❌ Error processing ZIP file: {str(e)}")
            return []
    
    # Option 2: Manual folder path (for advanced users)
    st.markdown("**Option 2: Enter folder path manually**")
    folder_path = st.text_input(
        "Enter the full path to your image folder:",
        placeholder="C:\\Users\\YourName\\Pictures\\FacialImages",
        help="Enter the complete path to the folder containing your images"
    )
    
    if folder_path and os.path.exists(folder_path):
        # Find all image files
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if image_files:
            st.success(f"✅ Found {len(image_files)} images in folder")
            return image_files
        else:
            st.warning("⚠️ No image files found in the specified folder")
            return []
    elif folder_path:
        st.error("❌ Folder path does not exist")
    
    return []


class FacialLabellingSystem:
    def __init__(self):
        self.age_gender_model = None
        self.models_loaded = False
        self.dataset_info = {}
        self.labeled_data = []
        self.unlabeled_data = []
        self.prediction_results = []
        
    def load_models(self):
        """Load the pre-trained models"""
        try:
            with st.spinner("🔄 Loading models..."):
                # Use MODEL_PATH from config
                model_path = os.path.abspath(MODEL_PATH)
                if os.path.exists(model_path):
                    self.age_gender_model = load_model(model_path, compile=False)
                    self.age_gender_model.compile(
                        optimizer="adam",
                        loss={
                            "age_out": "mse",
                            "sex_out": "binary_crossentropy"
                        },
                        metrics={
                            "age_out": "mae",
                            "sex_out": "accuracy"
                        }
                    )
                    self.models_loaded = True
                    return True
                else:
                    st.error(f"❌ Model file not found at: {model_path}")
                    st.error(f"Please ensure the model file exists in the labeling_age_gender directory.")
                    return False
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False

    
    def detect_and_crop_face(self, image_array):
        """Detect face in image and crop it.
        
        Args:
            image_array: numpy array of the image
            
        Returns:
            Cropped face image array, or original image if no face detected
        """
        if not ENABLE_FACE_DETECTION:
            return image_array
            
        try:
            # Convert to grayscale for face detection
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION_SCALE_FACTOR,
                minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
                minSize=FACE_DETECTION_MIN_SIZE
            )
            
            if len(faces) > 0:
                # Get the largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                
                # Add some padding to the face region
                padding = int(0.1 * max(w, h))
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image_array.shape[1] - x, w + 2 * padding)
                h = min(image_array.shape[0] - y, h + 2 * padding)
                
                # Crop the face
                face_crop = image_array[y:y+h, x:x+w]
                return face_crop
            else:
                # No face detected, return original image
                return image_array
                
        except Exception as e:
            # If face detection fails, return original image
            return image_array
    
    def preprocess_image(self, image, target_size=(128, 128)):
        """Preprocess image for model input
        
        Includes automatic face detection and cropping if enabled.
        """
        try:
            # Convert PIL image to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Detect and crop face if enabled
            img_array = self.detect_and_crop_face(img_array)
            
            # Resize to model input size
            img_resized = cv2.resize(img_array, target_size)
            
            # Normalize to [0,1]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None

    
    def predict_age_gender(self, image):
        """Predict age and gender from image"""
        try:
            # Preprocess image
            img_input = self.preprocess_image(image)
            if img_input is None:
                return None
            
            # Make prediction
            gender_pred, age_pred = self.age_gender_model.predict(img_input, verbose=0)
            
            # Process results
            gender_prob = float(gender_pred[0][0])
            gender_label = "Male" if gender_prob < 0.5 else "Female"
            gender_confidence = float(max(gender_prob, 1 - gender_prob))
            age_value = int(round(float(age_pred[0][0])))
            
            return {
                'age': age_value,
                'gender': gender_label,
                'gender_probability': gender_prob,
                'gender_confidence': gender_confidence,
                'raw_age': float(age_pred[0][0])
            }
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None

    def predict_all(self, image):
        """Prediction for age and gender"""
        try:
            age_gender_pred = self.predict_age_gender(image)
            if not age_gender_pred:
                return None
            
            result = {
                'age': age_gender_pred['age'],
                'gender': age_gender_pred['gender'],
                'gender_confidence': age_gender_pred['gender_confidence'],
                'raw_gender_prob': age_gender_pred['gender_probability']
            }
            
            return result
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None
    
    def process_dataset(self, labeled_files, unlabeled_files):
        """Process the entire dataset"""
        all_results = []
        
        # Process labeled data
        if labeled_files:
            st.info(f"📊 Processing {len(labeled_files)} labeled images...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(labeled_files):
                # Handle both file objects (uploaded files) and file paths (folder upload)
                filename = getattr(file_path, 'name', os.path.basename(str(file_path)))
                status_text.text(f"Processing labeled image {i+1}/{len(labeled_files)}: {filename}")
                
                try:
                    # Load image
                    image = Image.open(file_path)
                    
                    # Get filename info (assuming format: age_gender_*.jpg)
                    parts = filename.split('_')
                    
                    if len(parts) >= 2:
                        try:
                            true_age = int(parts[0])
                            true_gender = int(parts[1])
                            gender_label = "Male" if true_gender == 0 else "Female"
                            
                            # Make prediction
                            prediction = self.predict_age_gender(image)
                            
                            if prediction:
                                result = {
                                    'filename': filename,
                                    'image_type': 'Labeled',
                                    'true_age': true_age,
                                    'true_gender': gender_label,
                                    'predicted_age': prediction['age'],
                                    'predicted_gender': prediction['gender'],
                                    'age_error': abs(true_age - prediction['age']),
                                    'gender_correct': (true_gender == 0 and prediction['gender'] == 'Male') or 
                                                    (true_gender == 1 and prediction['gender'] == 'Female'),
                                    'gender_confidence': prediction['gender_confidence'],
                                    'raw_gender_prob': prediction['gender_probability']
                                }
                                all_results.append(result)
                        except ValueError:
                            st.warning(f"Could not parse filename: {filename}")
                    else:
                        st.warning(f"Invalid filename format: {filename}")
                        
                except Exception as e:
                    st.error(f"Error processing {file_path}: {str(e)}")
                    continue
                
                # Update progress
                progress_bar.progress((i + 1) / len(labeled_files))
            
            progress_bar.empty()
            status_text.empty()
        
        # Process unlabeled data
        if unlabeled_files:
            st.info(f"🔍 Processing {len(unlabeled_files)} unlabeled images...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(unlabeled_files):
                # Handle both file objects (uploaded files) and file paths (folder upload)
                filename = getattr(file_path, 'name', os.path.basename(str(file_path)))
                status_text.text(f"Processing unlabeled image {i+1}/{len(unlabeled_files)}: {filename}")
                
                try:
                    # Load image
                    image = Image.open(file_path)
                    
                    # Make prediction
                    prediction = self.predict_age_gender(image)
                    
                    if prediction:
                        result = {
                            'filename': filename,
                            'image_type': 'Unlabeled',
                            'true_age': None,
                            'true_gender': None,
                            'predicted_age': prediction['age'],
                            'predicted_gender': prediction['gender'],
                            'age_error': None,
                            'gender_correct': None,
                            'gender_confidence': prediction['gender_confidence'],
                            'raw_gender_prob': prediction['gender_probability']
                        }
                        all_results.append(result)
                        
                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")
                    continue
                
                # Update progress
                progress_bar.progress((i + 1) / len(unlabeled_files))
            
            progress_bar.empty()
            status_text.empty()
        
        return all_results

def upload_folder():
    """Create a folder upload interface"""
    st.markdown("### 📁 Upload Folder")
    
    # Option 1: Zip file upload
    st.markdown("**Option 1: Upload as ZIP file**")
    zip_file = st.file_uploader(
        "Upload a ZIP file containing your images",
        type=['zip'],
        help="Create a ZIP file with your images and upload it here"
    )
    
    if zip_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find all image files
                image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(temp_dir, '**', ext), recursive=True))
                
                if image_files:
                    st.success(f"✅ Found {len(image_files)} images in ZIP file")
                    return image_files
                else:
                    st.warning("⚠️ No image files found in ZIP file")
                    return []
        except Exception as e:
            st.error(f"❌ Error processing ZIP file: {str(e)}")
            return []
    
    # Option 2: Manual folder path (for advanced users)
    st.markdown("**Option 2: Enter folder path manually**")
    folder_path = st.text_input(
        "Enter the full path to your image folder:",
        placeholder="C:\\Users\\YourName\\Pictures\\FacialImages",
        help="Enter the complete path to the folder containing your images"
    )
    
    if folder_path and os.path.exists(folder_path):
        # Find all image files
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if image_files:
            st.success(f"✅ Found {len(image_files)} images in folder")
            return image_files
        else:
            st.warning("⚠️ No image files found in the specified folder")
            return []
    elif folder_path:
        st.error("❌ Folder path does not exist")
    
    return []

def show_load_dataset_page(facial_system):
    """Display the dataset loading page"""
    st.markdown('<h2 class="sub-header">📁 Load Dataset</h2>', unsafe_allow_html=True)
    
    if not facial_system.models_loaded:
        st.warning("⚠️ Please load the models first from the Home page.")
        return
    
    # Dataset upload section
    st.markdown("### 📤 Upload Your Dataset")
    
    # Upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["📁 Upload Folder (Recommended)", "📄 Upload Individual Files"],
        help="Select how you want to upload your images"
    )
    
    if upload_method == "📁 Upload Folder (Recommended)":
        st.markdown("""
        **Benefits of folder upload:**
        - Upload entire datasets at once
        - Faster than selecting individual files
        - Better for large collections
        - Maintains folder structure
        """)
        
        # Folder upload interface
        uploaded_files = upload_folder()
        
        if uploaded_files:
            # Separate labeled and unlabeled images
            labeled_files = []
            unlabeled_files = []
            
            for file_path in uploaded_files:
                # Handle both file objects and string paths
                filename = getattr(file_path, 'name', os.path.basename(str(file_path)))
                parts = filename.split('_')
                
                # Check if filename follows the labeled format (age_gender_*.jpg)
                if len(parts) >= 2:
                    try:
                        age = int(parts[0])
                        gender = int(parts[1])
                        if 0 <= age <= 100 and gender in [0, 1]:
                            labeled_files.append(file_path)
                            continue
                    except ValueError:
                        pass
                
                # If not labeled, consider it unlabeled
                unlabeled_files.append(file_path)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Labeled Images", len(labeled_files))
            with col2:
                st.metric("Unlabeled Images", len(unlabeled_files))
            
            # Store dataset info
            facial_system.dataset_info = {
                'labeled_count': len(labeled_files),
                'unlabeled_count': len(unlabeled_files),
                'total_count': len(uploaded_files),
                'labeled_files': labeled_files,
                'unlabeled_files': unlabeled_files
            }
            
            # Show sample images
            if labeled_files or unlabeled_files:
                st.markdown("### 🖼️ Sample Images")
                
                # Show labeled samples
                if labeled_files:
                    st.markdown("**Labeled Samples:**")
                    sample_labeled = labeled_files[:min(5, len(labeled_files))]
                    cols = st.columns(len(sample_labeled))
                    for i, (col, file_path) in enumerate(zip(cols, sample_labeled)):
                        with col:
                            try:
                                image = Image.open(file_path)
                                st.image(image, caption=f"Labeled {i+1}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                
                # Show unlabeled samples
                if unlabeled_files:
                    st.markdown("**Unlabeled Samples:**")
                    sample_unlabeled = unlabeled_files[:min(5, len(unlabeled_files))]
                    cols = st.columns(len(sample_unlabeled))
                    for i, (col, file_path) in enumerate(zip(cols, sample_unlabeled)):
                        with col:
                            try:
                                image = Image.open(file_path)
                                st.image(image, caption=f"Unlabeled {i+1}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
            
            st.success("✅ Dataset loaded successfully! You can now proceed to process the images.")
    
    else:
        # Individual file upload (original method)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏷️ Labeled Images")
            st.markdown("""
            Upload images that already have age and gender labels.
            
            **Expected filename format:** `age_gender_*.jpg`
            - `age`: Integer age (0-100)
            - `gender`: 0 for Male, 1 for Female
            - Example: `25_0_person1.jpg` (25-year-old male)
            """)
            
            labeled_files = st.file_uploader(
                "Choose labeled image files",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                key="labeled_uploader"
            )
        
        with col2:
            st.markdown("#### 🔍 Unlabeled Images")
            st.markdown("""
            Upload images without age and gender labels.
            
            The system will automatically predict:
            - Age (0-100 years)
            - Gender (Male/Female)
            - Confidence scores
            """)
            
            unlabeled_files = st.file_uploader(
                "Choose unlabeled image files",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                key="unlabeled_uploader"
            )
        
        # Dataset summary for individual files
        if labeled_files or unlabeled_files:
            st.markdown("---")
            st.markdown("### 📊 Dataset Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Labeled Images", len(labeled_files) if labeled_files else 0)
            
            with col2:
                st.metric("Unlabeled Images", len(unlabeled_files) if unlabeled_files else 0)
            
            with col3:
                total = (len(labeled_files) if labeled_files else 0) + (len(unlabeled_files) if unlabeled_files else 0)
                st.metric("Total Images", total)
            
            with col4:
                if labeled_files:
                    labeled_percentage = (len(labeled_files) / total) * 100
                    st.metric("Labeled %", f"{labeled_percentage:.1f}%")
            
            # Store dataset info
            facial_system.dataset_info = {
                'labeled_count': len(labeled_files) if labeled_files else 0,
                'unlabeled_count': len(unlabeled_files) if unlabeled_files else 0,
                'total_count': total,
                'labeled_files': labeled_files,
                'unlabeled_files': unlabeled_files
            }
            
            # Show sample images
            if labeled_files or unlabeled_files:
                st.markdown("### 🖼️ Sample Images")
                
                # Show labeled samples
                if labeled_files:
                    st.markdown("**Labeled Samples:**")
                    cols = st.columns(min(5, len(labeled_files)))
                    for i, col in enumerate(cols):
                        if i < len(labeled_files):
                            with col:
                                st.image(labeled_files[i], caption=f"Labeled {i+1}", use_column_width=True)
                
                # Show unlabeled samples
                if unlabeled_files:
                    st.markdown("**Unlabeled Samples:**")
                    cols = st.columns(min(5, len(unlabeled_files)))
                    for i, col in enumerate(cols):
                        if i < len(unlabeled_files):
                            with col:
                                st.image(unlabeled_files[i], caption=f"Unlabeled {i+1}", use_column_width=True)
            
            st.success("✅ Dataset loaded successfully! You can now proceed to process the images.")

def show_process_page(facial_system):
    """Display the image processing page"""
    st.markdown('<h2 class="sub-header">🔍 Process Images</h2>', unsafe_allow_html=True)
    
    if not facial_system.models_loaded:
        st.warning("⚠️ Please load the models first from the Home page.")
        return
    
    if not facial_system.dataset_info or facial_system.dataset_info['total_count'] == 0:
        st.warning("⚠️ Please load a dataset first from the 'Load Dataset' page.")
        return
    
    st.markdown("### 🚀 Start Processing")
    
    # Display dataset info in a nice format
    st.markdown("#### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", facial_system.dataset_info['total_count'])
    with col2:
        st.metric("Labeled Images", facial_system.dataset_info['labeled_count'])
    with col3:
        st.metric("Unlabeled Images", facial_system.dataset_info['unlabeled_count'])
    with col4:
        if facial_system.dataset_info['labeled_count'] > 0:
            labeled_percentage = (facial_system.dataset_info['labeled_count'] / facial_system.dataset_info['total_count']) * 100
            st.metric("Labeled %", f"{labeled_percentage:.1f}%")
    
    # Processing options
    st.markdown("#### ⚙️ Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of images to process at once. Lower values use less memory."
        )
    
    with col2:
        show_progress = st.checkbox(
            "Show detailed progress",
            value=True,
            help="Display real-time progress for each image"
        )
    
    # Process button
    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing images..."):
            # Process the dataset
            results = facial_system.process_dataset(
                facial_system.dataset_info.get('labeled_files', []),
                facial_system.dataset_info.get('unlabeled_files', [])
            )
            
            if results:
                facial_system.prediction_results = results
                st.success(f"✅ Processing complete! Processed {len(results)} images.")
                
                # Show processing summary
                show_processing_summary(results)
                
                # Store results in session state
                st.session_state.results = results
                st.rerun()
            else:
                st.error("❌ No results generated. Please check your dataset and try again.")

def show_processing_summary(results):
    """Display processing summary"""
    st.markdown("### 📊 Processing Summary")
    
    if not results:
        return
    
    # Convert to dataframe
    df = pd.DataFrame(results)
    
    # Calculate metrics
    labeled_results = df[df['image_type'] == 'Labeled']
    unlabeled_results = df[df['image_type'] == 'Unlabeled']
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Processed", len(results))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not labeled_results.empty:
            age_mae = labeled_results['age_error'].mean()
            st.metric("Age MAE (Labeled)", f"{age_mae:.1f}")
        else:
            st.metric("Age MAE (Labeled)", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not labeled_results.empty:
            gender_accuracy = labeled_results['gender_correct'].mean() * 100
            st.metric("Gender Accuracy", f"{gender_accuracy:.1f}%")
        else:
            st.metric("Gender Accuracy", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_confidence = df['gender_confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show detailed results table
    st.markdown("### 📋 Detailed Results")
    
    # Add search and filter options
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("🔍 Search by filename:", placeholder="Enter filename to search...")
    
    with col2:
        filter_type = st.selectbox("📁 Filter by type:", ["All", "Labeled", "Unlabeled"])
    
    # Filter data
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['filename'].str.contains(search_term, case=False, na=False)]
    if filter_type != "All":
        filtered_df = filtered_df[filtered_df['image_type'] == filter_type]
    
    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True)
    
    # Show summary statistics
    if not filtered_df.empty:
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} results**")

def show_results_page(facial_system):
    """Display the results page"""
    st.markdown('<h2 class="sub-header">📊 Results Analysis</h2>', unsafe_allow_html=True)
    
    if not facial_system.prediction_results:
        st.warning("⚠️ No results available. Please process images first.")
        return
    
    results = facial_system.prediction_results
    df = pd.DataFrame(results)
    
    # Results overview
    st.markdown("### 📈 Results Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        st.markdown("**Age Distribution**")
        fig_age = px.histogram(
            df, 
            x='predicted_age', 
            color='image_type',
            title="Predicted Age Distribution",
            nbins=20,
            color_discrete_map={'Labeled': '#1f77b4', 'Unlabeled': '#ff7f0e'}
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        st.markdown("**Gender Distribution**")
        gender_counts = df['predicted_gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Predicted Gender Distribution",
            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'}
        )
        fig_gender.update_layout(height=400)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Performance metrics for labeled data
    labeled_df = df[df['image_type'] == 'Labeled']
    if not labeled_df.empty:
        st.markdown("### 🎯 Performance Metrics (Labeled Data)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_mae = labeled_df['age_error'].mean()
            st.metric("Mean Age Error", f"{age_mae:.2f} years")
        
        with col2:
            gender_accuracy = labeled_df['gender_correct'].mean() * 100
            st.metric("Gender Accuracy", f"{gender_accuracy:.1f}%")
        
        with col3:
            avg_confidence = labeled_df['gender_confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Age error analysis
        st.markdown("**Age Prediction Error Analysis**")
        fig_age_error = px.scatter(
            labeled_df,
            x='true_age',
            y='predicted_age',
            title="True vs Predicted Age",
            labels={'true_age': 'True Age', 'predicted_age': 'Predicted Age'},
            color='gender_confidence',
            color_continuous_scale='viridis'
        )
        fig_age_error.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig_age_error, use_container_width=True)
    
    # Confidence analysis
    st.markdown("### 📊 Confidence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig_conf = px.histogram(
            df,
            x='gender_confidence',
            color='image_type',
            title="Gender Prediction Confidence Distribution",
            nbins=20,
            color_discrete_map={'Labeled': '#1f77b4', 'Unlabeled': '#ff7f0e'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Confidence vs accuracy (for labeled data)
        if not labeled_df.empty:
            fig_conf_acc = px.scatter(
                labeled_df,
                x='gender_confidence',
                y='gender_correct',
                title="Confidence vs Accuracy",
                labels={'gender_confidence': 'Confidence', 'gender_correct': 'Correct Prediction'},
                color='true_age',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_conf_acc, use_container_width=True)

def show_download_page(facial_system):
    """Display the download page"""
    st.markdown('<h2 class="sub-header">📥 Download Results</h2>', unsafe_allow_html=True)
    
    if not facial_system.prediction_results:
        st.warning("⚠️ No results available. Please process images first.")
        return
    
    results = facial_system.prediction_results
    df = pd.DataFrame(results)
    
    st.markdown("### 📊 Export Options")
    
    # Data preview with search and filter
    st.markdown("**Data Preview:**")
    
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("🔍 Search:", placeholder="Search by filename...")
    with col2:
        filter_type = st.selectbox("📁 Filter:", ["All", "Labeled", "Unlabeled"])
    
    # Filter data
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['filename'].str.contains(search_term, case=False, na=False)]
    if filter_type != "All":
        filtered_df = filtered_df[filtered_df['image_type'] == filter_type]
    
    st.dataframe(filtered_df.head(10), use_container_width=True)
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} results**")
    
    # Export options
    st.markdown("### 📁 Export Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 CSV Export")
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"facial_labelling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### 📊 Excel Export")
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metric': ['Total Images', 'Labeled Images', 'Unlabeled Images', 'Age MAE', 'Gender Accuracy'],
                'Value': [
                    len(filtered_df),
                    len(filtered_df[filtered_df['image_type'] == 'Labeled']),
                    len(filtered_df[filtered_df['image_type'] == 'Unlabeled']),
                    filtered_df[filtered_df['image_type'] == 'Labeled']['age_error'].mean() if len(filtered_df[filtered_df['image_type'] == 'Labeled']) > 0 else 'N/A',
                    f"{filtered_df[filtered_df['image_type'] == 'Labeled']['gender_correct'].mean() * 100:.1f}%" if len(filtered_df[filtered_df['image_type'] == 'Labeled']) > 0 else 'N/A'
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name=f"facial_labelling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Dataset statistics
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(filtered_df))
    
    with col2:
        labeled_count = len(filtered_df[filtered_df['image_type'] == 'Labeled'])
        st.metric("Labeled Images", labeled_count)
    
    with col3:
        unlabeled_count = len(filtered_df[filtered_df['image_type'] == 'Unlabeled'])
        st.metric("Unlabeled Images", unlabeled_count)
    
    with col4:
        if labeled_count > 0:
            accuracy = filtered_df[filtered_df['image_type'] == 'Labeled']['gender_correct'].mean() * 100
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Overall Accuracy", "N/A")
    
    # Additional export options
    st.markdown("### 🔧 Advanced Export Options")
    
    # Filtered exports
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export by Image Type:**")
        export_type = st.selectbox("Select image type:", ["All", "Labeled", "Unlabeled"], key="export_type")
        
        if export_type != "All":
            type_filtered_df = filtered_df[filtered_df['image_type'] == export_type]
        else:
            type_filtered_df = filtered_df
        
        if st.button(f"📥 Export {export_type} Images", use_container_width=True):
            csv_data = type_filtered_df.to_csv(index=False)
            st.download_button(
                label=f"Download {export_type} CSV",
                data=csv_data,
                file_name=f"facial_labelling_{export_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.markdown("**Export by Gender:**")
        gender_filter = st.selectbox("Select gender:", ["All", "Male", "Female"], key="gender_filter")
        
        if gender_filter != "All":
            gender_df = filtered_df[filtered_df['predicted_gender'] == gender_filter]
        else:
            gender_df = filtered_df
        
        if st.button(f"📥 Export {gender_filter} Results", use_container_width=True):
            csv_data = gender_df.to_csv(index=False)
            st.download_button(
                label=f"Download {gender_filter} CSV",
                data=csv_data,
                file_name=f"facial_labelling_{gender_filter.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    # Initialize the system
    if 'facial_system' not in st.session_state:
        st.session_state.facial_system = FacialLabellingSystem()
    
    facial_system = st.session_state.facial_system
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Automatic Facial Image Labelling System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
    
            ["🏠 Home", "🙍‍♀️ Single Image",  "📁 Load Dataset", "🔍 Process Images", "📊 Results", "📥 Download", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("## ℹ️ System Status")
        if facial_system.models_loaded:
            st.success("✅ Age/Gender Model Loaded")
        else:
            st.error("❌ Model Not Loaded")
        
        st.markdown("---")
        st.markdown("## 📊 Dataset Info")
        if facial_system.dataset_info:
            st.write(f"**Labeled Images:** {facial_system.dataset_info.get('labeled_count', 0)}")
            st.write(f"**Unlabeled Images:** {facial_system.dataset_info.get('unlabeled_count', 0)}")
            st.write(f"**Total Images:** {facial_system.dataset_info.get('total_count', 0)}")
    
    # Load models if not already loaded
    if not facial_system.models_loaded:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🚀 Get Started")
        st.markdown("Click the button below to load the deep learning models and begin using the system.")
        if st.button("🚀 Load Models", use_container_width=True):
            if facial_system.load_models():
                st.success("✅ Models loaded successfully!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🙍‍♀️ Single Image":
        show_single_image_page(facial_system)
    elif page == "📁 Load Dataset":
        show_load_dataset_page(facial_system)
    elif page == "🔍 Process Images":
        show_process_page(facial_system)
    elif page == "📊 Results":
        show_results_page(facial_system)
    elif page == "📥 Download":
        show_download_page(facial_system)
    elif page == "ℹ️ About":
        show_about_page()

def show_single_image_page(facial_system):
    """Display the single image analysis page"""
    st.markdown('<h2 class="sub-header">🖼️ Single Image Analysis</h2>', unsafe_allow_html=True)
    
    if not facial_system.models_loaded:
        st.warning("⚠️ Please load the models first from the Home page.")
        return
    
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a facial image for analysis"
    )
    
    if uploaded_file is not None:
        # Display the image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        
        with col2:
            st.markdown("#### 🔍 Analysis")
            
            if st.button("🚀 Analyze Image", use_container_width=True):
                with st.spinner("🔄 Processing image..."):
                    # Get comprehensive prediction
                    prediction = facial_system.predict_all(image)
                    
                    if prediction:
                        st.success("✅ Analysis complete!")
                        
                        # Display results in an organized way
                        st.markdown("##### 📊 Results")
                        
                        # Basic info
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Predicted Age", f"{prediction['age']} years")
                        with col_b:
                            st.metric("Predicted Gender", prediction['gender'])
                        
                        # Gender confidence
                        st.markdown("**Gender Confidence:**")
                        st.progress(float(prediction['gender_confidence']))
                        st.text(f"{float(prediction['gender_confidence']):.1%}")
                        
                        # Download individual result
                        st.markdown("##### 📥 Export Result")
                        result_data = {
                            'filename': uploaded_file.name,
                            'predicted_age': prediction['age'],
                            'predicted_gender': prediction['gender'],
                            'gender_confidence': prediction['gender_confidence'],
                            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        result_df = pd.DataFrame([result_data])
                        csv_data = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Result CSV",
                            data=csv_data,
                            file_name=f"single_image_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    else:
                        st.error("❌ Failed to analyze image. Please try another image.")
    
    else:
        st.info("👆 Please upload an image to begin analysis")
        
        # Show example of what the system can detect
        st.markdown("### 🎯 What This System Can Detect")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **👤 Age Prediction:**
            - Age (0-100 years)
            - Regression-based estimation
            - Continuous value output
            """)
        
        with col2:
            st.markdown("""
            **⚧️ Gender Classification:**
            - Binary classification (Male/Female)
            - Confidence scoring
            - Probability estimates
            """)

def show_home_page():
    """Display the home page - Clean and user-friendly"""
    st.markdown('<h2 class="sub-header">Welcome to Automatic Facial Image Labelling</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Automatically label facial images with **Age** and **Gender** predictions using advanced deep learning.
    """)
    
    st.markdown("---")
    
    # Step-by-step guide
    st.markdown('<h3 class="sub-header">📋 How to Use</h3>', unsafe_allow_html=True)
    
    # Step 1
    st.markdown("### Step 1️⃣ Initialize System")
    st.markdown("""
    Click the **Load Models** button in the sidebar to prepare the system. This loads the deep learning models needed for image analysis.
    """)
    st.info("⚠️ Models must be loaded before processing any images")
    
    # Step 2
    st.markdown("### Step 2️⃣ Choose Your Task")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Option A: Single Image**
        - Go to 🙍‍♀️ Single Image
        - Upload one image to get instant predictions
        - Perfect for quick testing
        """)
    with col2:
        st.markdown("""
        **Option B: Batch Processing**
        - Go to 📁 Load Dataset
        - Upload multiple images at once
        - Perfect for labeling large collections
        """)
    
    # Step 3
    st.markdown("### Step 3️⃣ Process & View Results")
    st.markdown("""
    - System automatically analyzes all images
    - View predictions (Age and Gender)
    - Check confidence scores for each prediction
    """)
    
    # Step 4
    st.markdown("### Step 4️⃣ Download Results")
    st.markdown("""
    - Go to 📥 Download
    - Export labeled data in CSV or Excel format
    - Ready to use for your projects
    """)
    
    st.markdown("---")
    
    # What it does
    st.markdown('<h3 class="sub-header">✨ What We Predict</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 👤 Age
        Estimated age in years
        (0-100 years)
        """)
    with col2:
        st.markdown("""
        ### 👫 Gender
        Male or Female
        with confidence score
        """)
    
    st.markdown("---")
    
    # Tips
    st.markdown('<h3 class="sub-header">💡 Tips for Best Results</h3>', unsafe_allow_html=True)
    st.markdown("""
    - **Good Lighting**: Clear, well-lit facial images work best
    - **Face Visibility**: Make sure faces are clearly visible
    - **Image Quality**: Higher resolution images give better results
    - **Multiple Images**: Batch processing is faster for large datasets
    
    ---
    **Want to know more?** Check the ℹ️ **About** page for technical details!
    """)

def show_about_page():
    """Display the about/info page with technical details"""
    st.markdown('<h2 class="sub-header">About This System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Overview
    
    The Automatic Facial Image Labelling System uses state-of-the-art deep learning models 
    to automatically identify and classify facial age and gender from images.
    
    ---
    
    ## 📊 Predictions Available
    
    **Age Prediction**
    - Regression-based estimation (0-100 years)
    - Mean Absolute Error (MAE) based accuracy
    
    **Gender Classification**
    - Binary classification (Male / Female)
    - Confidence scores for each prediction
    
    ---
    
    ## 🔬 Technical Details
    
    ### Model Architecture
    - **Age/Gender Model**: MobileNetV2 with dual output heads
    - **Input Size**: 128×128 pixels (RGB)
    - **Preprocessing**: Automatic face detection and cropping
    - **Normalization**: Image normalization and resizing
    - **Batch Processing**: Efficient multi-image processing
    
    ### Processing Pipeline
    1. Image loading and validation
    2. Automatic face detection using Haar Cascade
    3. Face region cropping and preprocessing
    4. Model inference with confidence scores
    5. Results aggregation and export
    
    ### File Formats
    - **Input**: JPG, PNG, BMP, TIFF
    - **Output**: CSV, Excel formats with predictions
    
    ---
    
    ## 🛠️ Features
    
    ✅ Single image analysis for quick testing  
    ✅ Batch processing for large datasets  
    ✅ Automatic face detection and cropping  
    ✅ Confidence scores for all predictions  
    ✅ Export results in CSV/Excel formats  
    ✅ Dataset statistics and visualization  
    
    ---
    
    ## 📈 Performance Metrics
    
    The system provides:
    - **Age accuracy** via Mean Absolute Error (MAE)
    - **Gender accuracy** with confidence percentages
    - **Processing statistics** (time, batch size, etc.)
    - **Dataset summaries** and error analysis
    
    ---
    
    ## 🔧 Configuration
    
    ### Model Setup
    The system automatically loads the trained model from:
    - `labeling_age_gender/Age_Gender_MobileNetV2.h5`
    - Detection parameter customization available in config.py
    
    ### Face Detection Settings
    - Scale Factor: 1.1
    - Min Neighbors: 5
    - Min Face Size: 30×30 pixels
    - Automatic padding around detected faces
    
    ### Requirements
    - TensorFlow/Keras for deep learning
    - OpenCV for image processing
    - Streamlit for the web interface
    - Pandas/Openpyxl for data export
    
    ---
    
    **Need help?** Check the Home page for step-by-step instructions!
    """)

if __name__ == "__main__":
    main()
