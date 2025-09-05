# Traffic Light Detection with OpenCV & Streamlit  

This project is a **real-time traffic light detection system** built with Python, OpenCV, and Streamlit.  
It uses **HSV color space segmentation** to detect and classify traffic light states (Red, Yellow, Green) from images, videos, or live webcam feed.  

## Features  
- Real-time detection of traffic light states  
- HSV color segmentation for robust detection  
- Bounding boxes and labels on detected lights  
- Streamlit web app interface with sliders to fine-tune HSV ranges  
- Works with webcam feed, video files, or uploaded images  

## Tech Stack  
- Python  
- OpenCV  
- NumPy  
- Streamlit  

## Getting Started  

### Windows  
```bat
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
