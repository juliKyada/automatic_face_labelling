# Configuration file for the Automatic Facial Image Labelling System

# GUI Settings
GUI_TITLE = "🤖 Automatic Facial Image Labelling System"
GUI_WIDTH = 1200
GUI_HEIGHT = 800
GUI_BACKGROUND = "#f0f0f0"
TITLE_BACKGROUND = "#2c3e50"
BUTTON_COLOR = "#3498db"
STATUS_BACKGROUND = "#ecf0f1"

# Model Settings
# Get the directory where this config file is located
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(_BASE_DIR, "Age_Gender_MobileNetV2.h5")

# Face Detection Settings
ENABLE_FACE_DETECTION = True  # Enable automatic face detection and cropping

# Image Settings
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
DISPLAY_SIZE = (400, 400)  # Max display size for images

# Face Detection Settings
ENABLE_FACE_DETECTION = True  # Enable automatic face detection and cropping
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (30, 30)  # Minimum face size to detect
IMAGE_QUALITY = 95  # JPEG quality for saving

# Processing Settings
BATCH_SIZE = 32  # Number of images to process at once
PROGRESS_UPDATE_INTERVAL = 100  # Update progress bar every N images

# Results Settings
EXPORT_FORMATS = ['.csv', '.xlsx']
DEFAULT_EXPORT_FORMAT = '.csv'

# UI Text
WELCOME_MESSAGE = """
Welcome to the Automatic Facial Image Labelling System!

• Upload an image to get started
• Use the Predict button to analyze the image
• Batch process multiple images at once
• View detailed results and export data

✅ Models are ready! You can now upload and analyze images.
"""

HELP_MESSAGE = """
How to use the system:

1. **Single Image Analysis:**
   - Click 'Upload Image' to select a file
   - Click 'Predict' to analyze
   - View results in the Single Image tab

2. **Batch Processing:**
   - Click 'Batch Process' to select a folder
   - Monitor progress with the progress bar
   - View results in the Batch Results tab
   - Export results using 'Export Results'

3. **Supported Formats:**
   - JPEG (.jpg, .jpeg)
   - PNG (.png)
   - BMP (.bmp)
   - TIFF (.tiff)

4. **Results:**
   - Age prediction (0-100 years)
   - Gender classification (Male/Female)
   - Confidence scores
   - Raw model outputs
"""

# Error Messages
ERROR_MODEL_NOT_FOUND = "Model file not found. Please ensure 'age_gender_mobilenetv2.h5' is in the current directory."
ERROR_IMAGE_LOAD = "Failed to load image. Please check file format and integrity."
ERROR_PREDICTION = "Prediction failed. Please try again or check the image."
ERROR_BATCH_PROCESS = "Batch processing failed. Please check the folder and try again."
ERROR_EXPORT = "Export failed. Please check file permissions and try again."

# Success Messages
SUCCESS_MODEL_LOAD = "✅ Models loaded successfully"
SUCCESS_PREDICTION = "Prediction completed successfully!"
SUCCESS_BATCH_COMPLETE = "Batch processing complete!"
SUCCESS_EXPORT = "Results exported successfully!"

# Warning Messages
WARNING_MODELS_LOADING = "Models are still loading. Please wait."
WARNING_NO_IMAGE = "Please upload an image first."
WARNING_NO_BATCH_RESULTS = "No batch results to export."

# File Paths (relative to project directory)
DEFAULT_EXPORT_DIR = "exports"
DEFAULT_BATCH_DIR = "batch_input"
DEFAULT_RESULTS_DIR = "results"

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FILE = "facial_labelling.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Performance Settings
ENABLE_MULTITHREADING = True
MAX_WORKERS = 4
MEMORY_LIMIT_MB = 2048  # 2GB memory limit for batch processing

# Display Settings
SHOW_CONFIDENCE_BARS = True
SHOW_RAW_VALUES = True
SHOW_PROCESSING_TIME = True
ENABLE_ANIMATIONS = True

# Export Settings
INCLUDE_TIMESTAMP = True
INCLUDE_MODEL_INFO = True
INCLUDE_PROCESSING_TIME = True
EXPORT_RAW_PREDICTIONS = False
