# 🤖 Automatic Facial Image Labelling System - Hugging Face Spaces Edition

A Streamlit-based application for automatic facial attribute detection including age, gender, and ethnicity predictions.

## Features

✨ **Real-time Predictions:**
- Age estimation
- Gender classification  
- Ethnicity detection

📤 **Batch Processing:**
- Upload multiple images
- Export results as Excel/CSV
- Visualizations and statistics

🎨 **User-Friendly Interface:**
- Drag-and-drop image uploads
- Real-time predictions
- Beautiful visualizations

## Model Information

- **Age/Gender Model**: `age_gender_mobilenetv2.h5`
- **Ethnicity Model**: `Ethnicity_lebelling.h5`

## How to Use

1. Upload an image using the interface
2. The system will automatically predict facial attributes
3. View results with confidence scores
4. Batch process multiple images for statistics

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run labeling_age_gender/streamlit_gui.py
```

## Deployment on Hugging Face Spaces

This app is deployed on Hugging Face Spaces using Streamlit. The app automatically loads models from the `labeling_age_gender` directory.

**Model Files Note:** Model files (`.h5`) should be placed in the `labeling_age_gender/` directory. They are excluded from Git LFS to keep the repository lightweight.

## File Structure

```
.
├── labeling_age_gender/
│   ├── streamlit_gui.py         # Main Streamlit app
│   ├── config.py                # Configuration
│   ├── age_gender_mobilenetv2.h5    # Age/Gender model
│   ├── Ethnicity_lebelling.h5       # Ethnicity model
│   └── ...
├── app.py                       # Entry point for HF Spaces
├── requirements.txt             # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+
- OpenCV, Pillow, and other dependencies (see requirements.txt)

## License

This project is available for use and modification.

## Support

For issues or questions, please check the GitHub repository or Hugging Face Spaces interface.
