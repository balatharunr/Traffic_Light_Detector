#!/usr/bin/env python3
"""
Streamlit App for Traffic Light Detection System

A web interface for the real-time traffic light detection system.
Users can upload videos or use webcam to detect traffic lights.
"""

# Robust OpenCV import with fallback - MUST be first
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    OPENCV_AVAILABLE = False
    cv2 = None

import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime

# Show error if OpenCV failed to import
if not OPENCV_AVAILABLE:
    st.error(
        "OpenCV import failed. Please ensure opencv-python-headless is installed."
    )
    st.error("Try: pip install opencv-python-headless")
    st.stop()

# Only import detector if OpenCV is available
if OPENCV_AVAILABLE:
    try:
        from traffic_light_detector import AutoTrafficLightDetector
        DETECTOR_AVAILABLE = True
    except ImportError as e:
        st.error(f"Traffic light detector import failed: {e}")
        DETECTOR_AVAILABLE = False
        AutoTrafficLightDetector = None
else:
    DETECTOR_AVAILABLE = False
    AutoTrafficLightDetector = None

# Page configuration
st.set_page_config(page_title="Traffic Light Detection System",
                   page_icon="🚦",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e3d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box p, .info-box li, .info-box h4 {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    .info-box h4 {
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
    }
    .detection-stats {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #c3e6cb;
    }
    .detection-stats p {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        margin: 0.5rem 0 !important;
    }
    .detection-stats h4 {
        color: #1f4e3d !important;
        font-weight: bold !important;
        margin-bottom: 1rem !important;
    }
    .section-header {
        color: #2c3e50;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #28a745;
    }
    .upload-area {
        border: 2px dashed #28a745;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #28a745;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .stSelectbox label, .sidebar .stSlider label, .sidebar .stCheckbox label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    .sample-button {
        margin: 0.5rem 0 !important;
        font-size: 0.9rem !important;
    }
    .sample-button > button {
        background-color: #6c757d !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.4rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s !important;
        width: 100% !important;
    }
    .sample-button > button:hover {
        background-color: #5a6268 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
</style>
""",
            unsafe_allow_html=True)


# Initialize the detector
@st.cache_resource
def get_detector():
    if DETECTOR_AVAILABLE and AutoTrafficLightDetector is not None:
        return AutoTrafficLightDetector()
    return None


detector = get_detector()


def create_debug_masks(image, detector):
    """Create debug mask visualization showing color segmentation"""
    if image is None:
        return None

    try:
        # Calculate dynamic parameters
        dynamic_params = detector._calculate_dynamic_params(image)

        # Preprocessing: white-balance, gamma, blur
        pre = detector._gray_world_white_balance(image)
        pre = detector._auto_gamma_correct(pre)
        blurred = cv2.GaussianBlur(pre, dynamic_params['blur_kernel_size'], 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])

        # Brightness and saturation gates
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        bright_thresh = max(20, min(60, int(np.percentile(gray, 15))))
        _, bright_mask = cv2.threshold(gray, bright_thresh, 255,
                                       cv2.THRESH_BINARY)

        if np.mean(s) < 15:
            combined_mask = bright_mask
        else:
            _, saturation_mask = cv2.threshold(s, 5, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_and(bright_mask, saturation_mask)

        # Create debug panel
        debug_panel = np.zeros((image.shape[0], image.shape[1], 3),
                               dtype=np.uint8)

        # Generate color masks for each color
        for color, ranges in detector.color_ranges.items():
            color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for r in ranges:
                mask = cv2.inRange(hsv, r['lower'], r['upper'])
                mask = cv2.bitwise_and(mask, combined_mask)
                color_mask = cv2.bitwise_or(color_mask, mask)

            # Apply morphology
            kernel_small = np.ones((dynamic_params['small_kernel_size'],
                                    dynamic_params['small_kernel_size']),
                                   np.uint8)
            kernel_close = np.ones(
                (max(7, dynamic_params['large_kernel_size']),
                 max(7, dynamic_params['large_kernel_size'])), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE,
                                          kernel_close)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,
                                          kernel_small)
            color_mask = cv2.medianBlur(color_mask, 3)

            # Color the mask
            if color == 'Red':
                color_bgr = (0, 0, 255)
            elif color == 'Yellow':
                color_bgr = (0, 255, 255)
            else:  # Green
                color_bgr = (0, 255, 0)

            colored = cv2.merge([
                (color_mask //
                 2) if color_bgr[0] > 0 else np.zeros_like(color_mask),
                (color_mask //
                 2) if color_bgr[1] > 0 else np.zeros_like(color_mask),
                (color_mask //
                 2) if color_bgr[2] > 0 else np.zeros_like(color_mask),
            ])
            debug_panel = cv2.add(debug_panel, colored)

        return debug_panel

    except Exception as e:
        st.error(f"Error creating debug masks: {str(e)}")
        return None


def process_image(image, filename):
    """Process uploaded image and display results"""

    if image is None:
        st.error("❌ Could not process image. Please try again.")
        return

    # Get image properties
    height, width = image.shape[:2]
    st.info(f"🖼️ Image Info: {width}x{height} pixels")

    # Create placeholders for image display
    image_placeholder = st.empty()
    status_text = st.empty()

    # Process image
    status_text.text("🔍 Processing image...")

    # Detect traffic lights once
    detections = detector.detect_traffic_lights(image)

    # Create annotated image manually to avoid double detection
    result_image = image.copy()
    for i, detection in enumerate(detections):
        x, y, w, h = detection['box']
        color_name = detection['color']
        confidence = detection['confidence']

        # Choose box color based on detected color
        if color_name == 'Red':
            box_color = (0, 0, 255)  # Red in BGR
        elif color_name == 'Yellow':
            box_color = (0, 255, 255)  # Yellow in BGR
        else:  # Green
            box_color = (0, 255, 0)  # Green in BGR

        # Draw bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), box_color, 3)

        # Draw label background
        cv2.rectangle(result_image, (x, y - 25), (x + w, y), box_color, -1)

        # Draw label text
        label = f"{color_name} ({confidence:.2f})"
        cv2.putText(result_image, label, (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add detection count info
    red_count = sum(1 for d in detections if d['color'] == 'Red')
    yellow_count = sum(1 for d in detections if d['color'] == 'Yellow')
    green_count = sum(1 for d in detections if d['color'] == 'Green')

    # Draw info bar
    info_height = 40
    overlay = result_image.copy()
    cv2.rectangle(overlay, (0, 0), (result_image.shape[1], info_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)

    # Add detection counts
    info_text = f"Red: {red_count} | Yellow: {yellow_count} | Green: {green_count} | Total: {len(detections)}"
    cv2.putText(result_image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # Initialize detection counts for this image
    current_detection_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}
    for detection in detections:
        color = detection.get('color', 'Unknown')
        if color in current_detection_counts:
            current_detection_counts[color] += 1

    # Update session state for sidebar stats - reset counts for each new image
    st.session_state.detection_counts = current_detection_counts.copy()

    # Convert BGR to RGB for display
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Display original and processed images side by side
    if show_debug:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📸 Original Image")
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, channels="RGB", use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Results")
            st.image(result_rgb, channels="RGB", use_container_width=True)

        with col3:
            st.subheader("🔍 Debug Masks")
            debug_image = create_debug_masks(image, detector)
            if debug_image is not None:
                debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                st.image(debug_rgb, channels="RGB", use_container_width=True)
            else:
                st.error("Could not generate debug masks")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Original Image")
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, channels="RGB", use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Results")
            st.image(result_rgb, channels="RGB", use_container_width=True)

    # Update status
    status_text.text(f"✅ Processing complete! | "
                     f"Red: {current_detection_counts['Red']} | "
                     f"Yellow: {current_detection_counts['Yellow']} | "
                     f"Green: {current_detection_counts['Green']}")

    # Display final statistics
    st.success("✅ Image processing complete!")

    # Display final statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Red Lights", current_detection_counts['Red'])
    with col2:
        st.metric("🟡 Yellow Lights", current_detection_counts['Yellow'])
    with col3:
        st.metric("🟢 Green Lights", current_detection_counts['Green'])

    # Show detection details
    if any(current_detection_counts.values()):
        st.subheader("🔍 Detection Details")
        for i, detection in enumerate(detections):
            x, y, w, h = detection['box']
            color = detection.get('color', 'Unknown')
            confidence = detection.get('confidence', 0)

            st.write(
                f"**{color} Light #{i+1}** - Confidence: {confidence:.2f} - Position: ({x}, {y}) - Size: {w}x{h}"
            )


def process_video(video_path):
    """Process uploaded video and display results"""

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Could not open video file. Please check the file format.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    st.info(
        f"📹 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration"
    )

    # Create placeholders for video display
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Detection counters
    detection_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}
    frame_count = 0

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result_frame = detector.process_frame(frame)

        # Count detections in current frame - use raw detections for better accuracy
        frame_detections = detector.detect_traffic_lights(frame)
        for detection in frame_detections:
            color = detection.get('color', 'Unknown')
            if color in detection_counts:
                detection_counts[color] += 1

        # Convert BGR to RGB for display
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # Display frame with or without debug masks
        if show_debug:
            debug_image = create_debug_masks(frame, detector)
            if debug_image is not None:
                debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                # Create side-by-side comparison
                combined = np.hstack([result_rgb, debug_rgb])
                video_placeholder.image(combined,
                                        channels="RGB",
                                        use_container_width=True)
            else:
                video_placeholder.image(result_rgb,
                                        channels="RGB",
                                        use_container_width=True)
        else:
            video_placeholder.image(result_rgb,
                                    channels="RGB",
                                    use_container_width=True)

        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        # Update status
        status_text.text(f"Processing frame {frame_count}/{total_frames} | "
                         f"Red: {detection_counts['Red']} | "
                         f"Yellow: {detection_counts['Yellow']} | "
                         f"Green: {detection_counts['Green']}")

        # Add small delay to make it viewable
        import time
        time.sleep(0.1)

    # Cleanup
    cap.release()

    # Update session state for sidebar stats - reset counts for each video
    st.session_state.detection_counts = detection_counts.copy()

    # Final results
    st.success("✅ Video processing complete!")

    # Display final statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Red Lights", detection_counts['Red'])
    with col2:
        st.metric("🟡 Yellow Lights", detection_counts['Yellow'])
    with col3:
        st.metric("🟢 Green Lights", detection_counts['Green'])

    # Clean up temporary file
    try:
        os.unlink(video_path)
    except:
        pass


# Check if dependencies are available
if not OPENCV_AVAILABLE or not DETECTOR_AVAILABLE or detector is None:
    st.error(
        "❌ Required dependencies are not available. Please check the installation."
    )
    st.stop()

# Main header
st.markdown('<h1 class="main-header">🚦 Traffic Light Detection System</h1>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Real-time detection of Red, Yellow, and Green traffic lights using OpenCV and HSV color segmentation</p>',
    unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.title(" Controls")

# Detection settings
st.sidebar.subheader("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Higher values = fewer but more confident detections")

show_debug = st.sidebar.checkbox("Show Debug Masks",
                                 value=False,
                                 help="Toggle color mask visualization")

# Update detector settings
detector.confidence_threshold = confidence_threshold
detector.show_debug_masks = show_debug

# Input method selection
st.sidebar.subheader("Input Method")
input_method = st.sidebar.radio(
    "Choose input source:", [
        "🖼️ Upload Image", "📹 Upload Video File", "📷 Use Webcam",
        "📁 Sample Images"
    ],
    help="Select how you want to provide input for traffic light detection")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Input")

    if input_method == "🖼️ Upload Image":
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to detect traffic lights")

        if uploaded_image is not None:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_image.read()),
                                    dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            st.success(f"✅ Image uploaded: {uploaded_image.name}")

            # Process image
            if st.button("🚀 Detect Traffic Lights", type="primary"):
                process_image(image, uploaded_image.name)

    elif input_method == "📹 Upload Video File":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Upload a video file to detect traffic lights")

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f'.{uploaded_file.name.split(".")[-1]}'
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            st.success(f"✅ Video uploaded: {uploaded_file.name}")

            # Process video
            if st.button("🚀 Start Detection", type="primary"):
                process_video(video_path)

    elif input_method == "📷 Use Webcam":
        st.info(
            "💡 Webcam functionality requires running locally. For web deployment, please use image or video upload."
        )

        if st.button("📷 Start Webcam Detection", type="primary"):
            st.warning(
                "⚠️ Webcam access is not available in Streamlit Cloud. Please run locally or upload an image/video file."
            )

    elif input_method == "📁 Sample Images":
        st.info(
            "🎯 Choose from our sample images to test the detection system:")

        # Create columns for sample images
        col1, col2, col3 = st.columns(3)

        sample_images = {
            "🔴 Red Light": "sample_images/sample_red_light.jpg",
            "🟡 Yellow Light": "sample_images/sample_yellow_light.jpg",
            "🟢 Green Light": "sample_images/sample_green_light.jpg",
            "🚦 All Lights": "sample_images/sample_all_lights.jpg",
            "🏙️ Multiple Lights": "sample_images/sample_multiple_lights.jpg",
            "🌙 Night Scene": "sample_images/sample_night_scene.jpg",
            "🎯 Challenging Scene": "sample_images/sample_challenging_scene.jpg"
        }

        # Initialize selected sample in session state
        if 'selected_sample' not in st.session_state:
            st.session_state.selected_sample = None

        with col1:
            st.markdown('<div class="sample-button">', unsafe_allow_html=True)
            if st.button("🔴 Red Light", use_container_width=True):
                st.session_state.selected_sample = "sample_images/sample_red_light.jpg"
                st.session_state.auto_scroll = True
                st.rerun()
            if st.button("🟡 Yellow Light", use_container_width=True):
                st.session_state.selected_sample = "sample_images/sample_yellow_light.jpg"
                st.session_state.auto_scroll = True
                st.rerun()
            if st.button("🟢 Green Light", use_container_width=True):
                st.session_state.selected_sample = "sample_images/sample_green_light.jpg"
                st.session_state.auto_scroll = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="sample-button">', unsafe_allow_html=True)
            if st.button("🚦 All Lights", use_container_width=True):
                st.session_state.selected_sample = "sample_images/sample_all_lights.jpg"
                st.session_state.auto_scroll = True
                st.rerun()
            if st.button("🏙️ Multiple Lights", use_container_width=True):
                st.session_state.selected_sample = "sample_images/sample_multiple_lights.jpg"
                st.session_state.auto_scroll = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="sample-button">', unsafe_allow_html=True)
            if st.button("🌙 Night Scene", use_container_width=True):
                st.session_state.selected_sample = "sample_images/sample_night_scene.jpg"
                st.session_state.auto_scroll = True
                st.rerun()
            if st.button("🎯 Challenging", use_container_width=True):
                st.session_state.selected_sample = "sample_images/sample_challenging_scene.jpg"
                st.session_state.auto_scroll = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Add instruction when sample is selected
        if st.session_state.selected_sample:
            st.info(
                "📝 **Please scroll down to see the detect button and process your selected image!**"
            )
            st.markdown("---")

        if st.session_state.selected_sample:
            # Debug: Show current working directory and file path
            current_dir = os.getcwd()
            full_path = os.path.abspath(st.session_state.selected_sample)

            # Try different possible paths
            possible_paths = [
                st.session_state.selected_sample,  # Relative path
                os.path.join(current_dir, st.session_state.selected_sample
                             ),  # Current dir + relative
                os.path.join(
                    os.path.dirname(__file__),
                    st.session_state.selected_sample),  # Script dir + relative
            ]

            image_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break

            if image_path:
                # Load and display the selected sample image
                image = cv2.imread(image_path)
                if image is not None:
                    st.success(f"✅ Selected: {os.path.basename(image_path)}")

                    # Add instruction
                    st.info(
                        "💡 Click the button below to detect traffic lights in this image"
                    )

                    # Display the sample image
                    if show_debug:
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                     channels="RGB",
                                     use_container_width=True,
                                     caption=
                                     f"Sample: {os.path.basename(image_path)}")
                        with col_img2:
                            debug_image = create_debug_masks(image, detector)
                            if debug_image is not None:
                                debug_rgb = cv2.cvtColor(
                                    debug_image, cv2.COLOR_BGR2RGB)
                                st.image(debug_rgb,
                                         channels="RGB",
                                         use_container_width=True,
                                         caption="Debug Masks")
                            else:
                                st.error("Could not generate debug masks")
                    else:
                        st.image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True,
                            caption=f"Sample: {os.path.basename(image_path)}")

                    # Process the sample image
                    # Create a container with ID for auto-scrolling
                    st.markdown('<div id="detect-button-section"></div>',
                                unsafe_allow_html=True)

                    # Add a simple header for the detect section
                    st.markdown("### 🚀 Ready to Detect Traffic Lights")

                    detect_container = st.container()
                    with detect_container:
                        col_btn1, col_btn2 = st.columns([2, 1])
                        with col_btn1:
                            if st.button("🚀 Detect Traffic Lights in Sample",
                                         type="primary"):
                                process_image(image,
                                              os.path.basename(image_path))
                        with col_btn2:
                            if st.button("❌ Clear Selection"):
                                st.session_state.selected_sample = None
                                st.rerun()

                    # Reset the auto_scroll flag
                    if st.session_state.get('auto_scroll', False):
                        st.session_state.auto_scroll = False
                else:
                    st.error("❌ Could not load the selected sample image.")
            else:
                st.warning(
                    f"⚠️ Sample images not found. Current directory: {current_dir}"
                )
                st.warning(f"Looking for: {st.session_state.selected_sample}")
                st.warning(f"Tried paths: {possible_paths}")
                st.info(
                    "💡 Make sure you're running the app from the project root directory."
                )

with col2:
    st.subheader("📊 Detection Info")

    # Info box
    st.markdown("""
    <div class="info-box">
        <h4>🔍 How it works:</h4>
        <ul>
            <li>HSV color segmentation</li>
            <li>Adaptive preprocessing</li>
            <li>Contour filtering</li>
            <li>Tracking & smoothing</li>
        </ul>
        <br>
        <h4>📸 Supported formats:</h4>
        <ul>
            <li><strong>Images:</strong> JPG, PNG, BMP, TIFF</li>
            <li><strong>Videos:</strong> MP4, AVI, MOV, MKV</li>
        </ul>
    </div>
    """,
                unsafe_allow_html=True)

    # Detection stats - dynamic based on session state
    if 'detection_counts' not in st.session_state:
        st.session_state.detection_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}

    # Use Streamlit native components for better reactivity
    st.markdown("""
    <div class="detection-stats">
        <h4>📈 Detection Stats:</h4>
    </div>
    """,
                unsafe_allow_html=True)

    # Check if any detection has been run
    total_detections = sum(st.session_state.detection_counts.values())
    if total_detections == 0:
        st.info(
            "ℹ️ No detections yet. Select an image and click 'Detect Traffic Lights' to see results."
        )

    # Display stats using Streamlit metrics for better reactivity
    col_red, col_yellow, col_green = st.columns(3)
    with col_red:
        st.metric("🔴 Red", st.session_state.detection_counts['Red'])
    with col_yellow:
        st.metric("🟡 Yellow", st.session_state.detection_counts['Yellow'])
    with col_green:
        st.metric("🟢 Green", st.session_state.detection_counts['Green'])

    # Debug info
    if st.checkbox("Show Debug Info"):
        st.write("Session state detection_counts:",
                 st.session_state.detection_counts)
        st.write("Session state keys:", list(st.session_state.keys()))
        if 'selected_sample' in st.session_state:
            st.write("Selected sample:", st.session_state.selected_sample)

    # Clear stats button
    if st.button("🔄 Clear Stats", use_container_width=True):
        st.session_state.detection_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚦 Traffic Light Detection System | Built with OpenCV & Streamlit</p>
    <p>Real-time HSV-based color detection with adaptive preprocessing</p>
</div>
""",
            unsafe_allow_html=True)

# Instructions for local testing
if st.sidebar.checkbox("Show Local Testing Instructions"):
    st.sidebar.markdown("""
    **To test locally:**
    1. Install: `pip install streamlit`
    2. Run: `streamlit run app.py`
    3. Open: http://localhost:8501
    """)

# Instructions for deployment
if st.sidebar.checkbox("Show Deployment Instructions"):
    st.sidebar.markdown("""
    **To deploy to Streamlit Cloud:**
    1. Push code to GitHub
    2. Go to share.streamlit.io
    3. Connect your repo
    4. Deploy!
    """)
