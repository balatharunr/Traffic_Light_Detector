#!/usr/bin/env python3
"""
Enhanced Streamlit App for Traffic Light Detection System

A web interface for real-time traffic light detection with improved controls and flexibility.
Users can upload images/videos, use webcam feeds, and adjust detection parameters.
"""
import streamlit as st
import numpy as np
import tempfile
import os
import cv2
import time
from datetime import datetime
from traffic_light_detector import AutoTrafficLightDetector

# Page configuration
st.set_page_config(page_title="Traffic Light Detection System",
                   page_icon="🚦",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e3d;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    .detected-light {
        padding: 5px 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .red-light {
        background-color: rgba(255, 0, 0, 0.2);
        border: 1px solid red;
    }
    .yellow-light {
        background-color: rgba(255, 255, 0, 0.2);
        border: 1px solid #FFD700;
    }
    .green-light {
        background-color: rgba(0, 255, 0, 0.2);
        border: 1px solid green;
    }
    .webcam-container {
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize detector
@st.cache_resource
def get_detector():
    return AutoTrafficLightDetector()

detector = get_detector()

# Initialize session state variables
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'stop_webcam' not in st.session_state:
    st.session_state.stop_webcam = False
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'has_captured_image' not in st.session_state:
    st.session_state.has_captured_image = False
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = {
        'Red': 0,
        'Yellow': 0,
        'Green': 0,
        'detections': []
    }

# Webcam control helpers (non-blocking streaming)
def start_webcam():
    """Initialize webcam and mark active."""
    try:
        if 'cap' in st.session_state and getattr(st.session_state.cap, 'isOpened', lambda: False)():
            st.session_state.webcam_active = True
            st.session_state.stop_webcam = False
            return
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("Failed to open webcam. Please check your camera connection.")
            st.session_state.webcam_active = False
            return
        st.session_state.webcam_active = True
        st.session_state.stop_webcam = False
    except Exception as e:
        st.error(f"Error accessing webcam: {e}")
        st.session_state.webcam_active = False

def toggle_webcam():
    """Toggle webcam on/off based on current state"""
    if st.session_state.webcam_active:
        stop_webcam()
    else:
        start_webcam()
if 'hsv_ranges' not in st.session_state:
    st.session_state.hsv_ranges = {
        'Red': [
            {
                'lower': [0, 20, 100],
                'upper': [10, 255, 255]
            },
            {
                'lower': [170, 20, 100],
                'upper': [180, 255, 255]
            },
        ],
        'Yellow': [
            {
                'lower': [20, 20, 100],
                'upper': [40, 255, 255]
            },
        ],
        'Green': [
            {
                'lower': [45, 20, 100],
                'upper': [75, 255, 255]
            },
        ]
    }

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

def process_image(image):
    """Process a single image for traffic light detection"""
    if image is None:
        return None, []
    
    # Update detector with HSV ranges from sidebar controls
    update_detector_hsv_ranges()
    
    # Process image
    detections = detector.detect_traffic_lights(image)
    
    # Create annotated image
    result_image = image.copy()
    
    # Update detection counts
    st.session_state.detection_results = {
        'Red': 0,
        'Yellow': 0,
        'Green': 0,
        'detections': []
    }
    
    for detection in detections:
        x, y, w, h = detection['box']
        color_name = detection['color']
        confidence = detection['confidence']
        
        # Update counts
        st.session_state.detection_results[color_name] += 1
        st.session_state.detection_results['detections'].append({
            'color': color_name,
            'confidence': confidence,
            'position': (x, y),
            'size': (w, h)
        })
        
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
    
    # Add info bar with detection counts
    info_height = 40
    overlay = result_image.copy()
    cv2.rectangle(overlay, (0, 0), (result_image.shape[1], info_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
    
    # Add detection counts
    info_text = f"Red: {st.session_state.detection_results['Red']} | Yellow: {st.session_state.detection_results['Yellow']} | Green: {st.session_state.detection_results['Green']} | Total: {len(detections)}"
    cv2.putText(result_image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    
    return result_image, detections

def update_detector_hsv_ranges():
    """Use default HSV ranges for detection"""
    # The HSV values are already set in the session state, so we just use them
    detector.color_ranges = {
        'Red': [
            {
                'lower': np.array([st.session_state.red_h_min1, st.session_state.red_s_min, st.session_state.red_v_min]),
                'upper': np.array([st.session_state.red_h_max1, st.session_state.red_s_max, st.session_state.red_v_max])
            },
            {
                'lower': np.array([st.session_state.red_h_min2, st.session_state.red_s_min, st.session_state.red_v_min]),
                'upper': np.array([st.session_state.red_h_max2, st.session_state.red_s_max, st.session_state.red_v_max])
            }
        ],
        'Yellow': [
            {
                'lower': np.array([st.session_state.yellow_h_min, st.session_state.yellow_s_min, st.session_state.yellow_v_min]),
                'upper': np.array([st.session_state.yellow_h_max, st.session_state.yellow_s_max, st.session_state.yellow_v_max])
            }
        ],
        'Green': [
            {
                'lower': np.array([st.session_state.green_h_min, st.session_state.green_s_min, st.session_state.green_v_min]),
                'upper': np.array([st.session_state.green_h_max, st.session_state.green_s_max, st.session_state.green_v_max])
            }
        ]
    }

def process_video_frame(frame):
    """Process a single video frame"""
    if frame is None:
        return None
    
    # Update detector with current HSV ranges
    update_detector_hsv_ranges()
    
    # Process the frame
    result_frame = detector.process_frame(frame)
    return result_frame

def _render_webcam_frame_once():
    """Read, process, and render a single webcam frame (non-blocking)."""
    if 'cap' not in st.session_state:
        return
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Failed to capture frame from webcam.")
        stop_webcam()
        return
    start_time = time.time()
    result_frame = process_video_frame(frame)
    fps = 1.0 / max(1e-6, (time.time() - start_time))
    cv2.putText(result_frame, f"FPS: {fps:.1f}", (result_frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    # Ensure placeholder exists
    if 'webcam_placeholder' not in st.session_state:
        st.session_state.webcam_placeholder = st.empty()
    st.session_state.webcam_placeholder.image(result_rgb, channels="RGB", use_container_width=True)

def stop_webcam():
    """Stop the webcam processing loop"""
    st.session_state.stop_webcam = True
    st.session_state.webcam_active = False
    if 'cap' in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap
    # Clear live view placeholder when stopping
    if 'webcam_placeholder' in st.session_state:
        st.session_state.webcam_placeholder.empty()
        
def delete_captured_image():
    """Delete the captured image and related data"""
    if 'current_frame' in st.session_state:
        del st.session_state.current_frame
    if 'result_image' in st.session_state:
        del st.session_state.result_image
    if 'detections' in st.session_state:
        del st.session_state.detections
    st.session_state.has_captured_image = False

def display_detection_info(detections):
    """Display detailed information about detections"""
    if not detections:
        st.info("No traffic lights detected.")
        return
    
    for i, detection in enumerate(detections):
        color_name = detection['color']
        confidence = detection['confidence']
        x, y, w, h = detection['box']
        
        css_class = ""
        if color_name == 'Red':
            css_class = "red-light"
        elif color_name == 'Yellow':
            css_class = "yellow-light"
        else:  # Green
            css_class = "green-light"
        
        st.markdown(f"""
        <div class="detected-light {css_class}">
            {color_name} light #{i+1} - Confidence: {confidence:.2f}
            <br>Position: ({x}, {y}) - Size: {w}x{h}
        </div>
        """, unsafe_allow_html=True)

# Main app header
st.markdown('<h1 class="main-header">🚦 Traffic Light Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect Red, Yellow, and Green traffic lights automatically</p>', unsafe_allow_html=True)

# Create sidebar for controls
st.sidebar.title(" Detection Controls")

# Set default HSV values for traffic light detection
if 'red_h_min1' not in st.session_state:
    # First red range (0-15) - expanded range for better red detection
    st.session_state.red_h_min1 = 0
    st.session_state.red_h_max1 = 15
    # Second red range (165-180) - expanded range for better red detection
    st.session_state.red_h_min2 = 165
    st.session_state.red_h_max2 = 180
    # Saturation and Value (same for both ranges) - lower saturation min for better detection
    st.session_state.red_s_min = 15
    st.session_state.red_s_max = 255
    st.session_state.red_v_min = 90
    st.session_state.red_v_max = 255

# Yellow light HSV values - narrowed range to prevent misclassification
if 'yellow_h_min' not in st.session_state:
    st.session_state.yellow_h_min = 22
    st.session_state.yellow_h_max = 38
    st.session_state.yellow_s_min = 20
    st.session_state.yellow_s_max = 255
    st.session_state.yellow_v_min = 100
    st.session_state.yellow_v_max = 255

# Green light HSV values
if 'green_h_min' not in st.session_state:
    st.session_state.green_h_min = 45
    st.session_state.green_h_max = 75
    st.session_state.green_s_min = 20
    st.session_state.green_s_max = 255
    st.session_state.green_v_min = 100
    st.session_state.green_v_max = 255

# Detection settings
st.sidebar.subheader("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
show_debug = st.sidebar.checkbox("Show Debug Masks", value=False)

# Update detector settings
detector.confidence_threshold = confidence_threshold
detector.show_debug_masks = show_debug

# Main content area - Input selection tabs
tab1, tab2, tab3, tab4 = st.tabs(["📷 Webcam", "🖼️ Upload Image", "📹 Upload Video", "📁 Sample Images"])

# Tab 1: Webcam Input
with tab1:
    st.header("📷 Live Webcam Detection")
    
    # Webcam control button (toggle between start and stop)
    webcam_button_label = "Stop Webcam" if st.session_state.webcam_active else "Start Webcam"
    if st.button(webcam_button_label, key="webcam_toggle_btn", use_container_width=True):
        toggle_webcam()
    
    # Capture frame and Delete buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        # Capture frame button (only enabled when webcam is active)
        if st.button("Capture Frame", key="capture_frame_btn", disabled=not st.session_state.webcam_active, use_container_width=True):
            if 'cap' in st.session_state and st.session_state.webcam_active:
                ret, frame = st.session_state.cap.read()
                if ret:
                    st.session_state.current_frame = frame
                    result_image, detections = process_image(frame)
                    st.session_state.result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.session_state.detections = detections
                    st.session_state.has_captured_image = True
                    st.success("Frame captured and processed!")
    
    with col2:
        # Delete captured image button (only enabled when there's a captured image)
        has_image = 'result_image' in st.session_state and 'current_frame' in st.session_state
        if st.button("Delete Captured Image", key="delete_image_btn", disabled=not has_image, use_container_width=True):
            delete_captured_image()
            st.success("Captured image deleted successfully!")
    
    # Create webcam placeholder container and render one frame if active
    webcam_container = st.container()
    with webcam_container:
        if not st.session_state.webcam_active:
            st.markdown('<div class="webcam-container" style="min-height: 300px; display: flex; align-items: center; justify-content: center;"><p style="color: #666;">Webcam inactive. Click the button above to start.</p></div>', unsafe_allow_html=True)
        else:
            # Ensure placeholder exists
            if 'webcam_placeholder' not in st.session_state:
                st.session_state.webcam_placeholder = st.empty()
            # Render one frame this run
            _render_webcam_frame_once()
    
    # Display captured frame and results if available
    if not st.session_state.webcam_active and 'result_image' in st.session_state and 'current_frame' in st.session_state:
        st.subheader("📸 Captured Frame Result")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB), caption="Original Frame", use_container_width=True)
        with col2:
            st.image(st.session_state.result_image, caption="Detection Result", use_container_width=True)
        
        # Show detection statistics
        st.subheader("🔍 Detection Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Red Lights", st.session_state.detection_results['Red'])
        with col2:
            st.metric("🟡 Yellow Lights", st.session_state.detection_results['Yellow'])
        with col3:
            st.metric("🟢 Green Lights", st.session_state.detection_results['Green'])
        
        # Display detailed detection information
        display_detection_info(st.session_state.detections)

    # Keep updating the live stream while active without blocking the UI
    if st.session_state.webcam_active:
        time.sleep(0.03)
        # Support both modern and legacy Streamlit versions
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

# Tab 2: Image Upload
with tab2:
    st.header("🖼️ Image Upload")
    
    # Store uploaded image in session state
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'uploaded_result_image' not in st.session_state:
        st.session_state.uploaded_result_image = None
    if 'uploaded_detections' not in st.session_state:
        st.session_state.uploaded_detections = None
    if 'uploaded_debug_image' not in st.session_state:
        st.session_state.uploaded_debug_image = None
    
    # Function to delete uploaded image results
    def delete_uploaded_image():
        st.session_state.uploaded_image = None
        st.session_state.uploaded_result_image = None
        st.session_state.uploaded_detections = None
        st.session_state.uploaded_debug_image = None
        
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None and st.button("Detect Traffic Lights", key="detect_uploaded_img"):
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Process image
            result_image, detections = process_image(image)
            
            # Store results in session state
            st.session_state.uploaded_image = image
            st.session_state.uploaded_result_image = result_image
            st.session_state.uploaded_detections = detections
            
            # Create debug image if enabled
            if show_debug:
                st.session_state.uploaded_debug_image = create_debug_masks(image, detector)
            
            st.success("Image processed successfully!")
            
    with col2:
        # Delete uploaded image results
        if st.session_state.uploaded_image is not None and st.button("Delete Results", key="delete_uploaded_img", use_container_width=True):
            delete_uploaded_image()
            st.success("Image results deleted successfully!")
    
    # Display results if available
    if st.session_state.uploaded_image is not None:
        # Display results
        if show_debug and st.session_state.uploaded_debug_image is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(st.session_state.uploaded_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(st.session_state.uploaded_result_image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
            with col3:
                if st.session_state.uploaded_debug_image is not None:
                    st.image(cv2.cvtColor(st.session_state.uploaded_debug_image, cv2.COLOR_BGR2RGB), caption="Debug Masks", use_container_width=True)
                else:
                    st.error("Could not generate debug masks")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(st.session_state.uploaded_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(st.session_state.uploaded_result_image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
        
        # Show detection statistics
        st.subheader("🔍 Detection Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Red Lights", st.session_state.detection_results['Red'])
        with col2:
            st.metric("🟡 Yellow Lights", st.session_state.detection_results['Yellow'])
        with col3:
            st.metric("🟢 Green Lights", st.session_state.detection_results['Green'])
        
        # Display detailed detection information
        display_detection_info(st.session_state.uploaded_detections)

# Tab 3: Video Upload
with tab3:
    st.header("📹 Video Upload")
    
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_video is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_video.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        st.success(f"Video uploaded: {uploaded_video.name}")
        
        if st.button("Process Video"):
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Failed to open video file.")
            else:
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.info(f"Video info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
                
                # Create placeholders
                video_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video frames
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    result_frame = process_video_frame(frame)
                    
                    # Convert BGR to RGB for display
                    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    video_placeholder.image(result_rgb, channels="RGB", use_container_width=True)
                    
                    # Update progress
                    frame_count += 1
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    
                    # Update status
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    # Limit display rate to make it viewable
                    time.sleep(0.1)
                
                # Clean up
                cap.release()
                
                # Delete temporary file
                try:
                    os.unlink(video_path)
                except:
                    pass
                
                status_text.text("Video processing complete!")

# Tab 4: Sample Images
with tab4:
    st.header("📁 Sample Images")
    
    # Initialize sample image session states
    if 'sample_image' not in st.session_state:
        st.session_state.sample_image = None
    if 'sample_result_image' not in st.session_state:
        st.session_state.sample_result_image = None
    if 'sample_detections' not in st.session_state:
        st.session_state.sample_detections = None
    if 'sample_debug_image' not in st.session_state:
        st.session_state.sample_debug_image = None
    
    # Function to delete sample image results
    def delete_sample_image_results():
        st.session_state.sample_result_image = None
        st.session_state.sample_detections = None
        st.session_state.sample_debug_image = None
    
    col1, col2 = st.columns(2)
    
    # Sample image selection
    with col1:
        st.subheader("Choose a Sample Image")
        sample_images = {
            "🔴 Red Light": "sample_images/sample_red_light.jpg",
            "🟡 Yellow Light": "sample_images/sample_yellow_light.jpg",
            "🟢 Green Light": "sample_images/sample_green_light.jpg",
            "🚦 All Lights": "sample_images/sample_all_lights.jpg",
            "🏙️ Multiple Lights": "sample_images/sample_multiple_lights.jpg",
            "🌙 Night Scene": "sample_images/sample_night_scene.jpg",
            "🎯 Challenging Scene": "sample_images/sample_challenging_scene.jpg"
        }
        
        selected_sample = st.selectbox("Select a sample image", list(sample_images.keys()))
        sample_path = sample_images[selected_sample]
        
        # Load the selected sample
        if os.path.exists(sample_path):
            sample_image = cv2.imread(sample_path)
            if sample_image is not None:
                st.session_state.sample_image = sample_image
                st.image(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB), caption=f"Sample: {selected_sample}", use_container_width=True)
                
                # Action buttons
                col_detect, col_delete = st.columns(2)
                with col_detect:
                    if st.button("Detect Traffic Lights", key="detect_sample_btn"):
                        # Process the sample image
                        result_image, detections = process_image(sample_image)
                        st.session_state.sample_result_image = result_image
                        st.session_state.sample_detections = detections
                        
                        # Create debug image if needed
                        if show_debug:
                            st.session_state.sample_debug_image = create_debug_masks(sample_image, detector)
                        
                        st.success("Sample image processed successfully!")
                
                with col_delete:
                    # Only show delete button if we have results
                    if st.session_state.sample_result_image is not None and st.button("Delete Results", key="delete_sample_btn"):
                        delete_sample_image_results()
                        st.success("Sample image results deleted!")
                
                # Display results if available
                if st.session_state.sample_result_image is not None:
                    # Display the result
                    if show_debug and st.session_state.sample_debug_image is not None:
                        col_orig, col_result, col_debug = st.columns(3)
                        with col_orig:
                            st.image(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
                        with col_result:
                            st.image(cv2.cvtColor(st.session_state.sample_result_image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
                        with col_debug:
                            st.image(cv2.cvtColor(st.session_state.sample_debug_image, cv2.COLOR_BGR2RGB), caption="Debug Masks", use_container_width=True)
                    else:
                        col_orig, col_result = st.columns(2)
                        with col_orig:
                            st.image(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
                        with col_result:
                            st.image(cv2.cvtColor(st.session_state.sample_result_image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
                    
                    # Show detection statistics
                    st.subheader("🔍 Detection Results")
                    col_r, col_y, col_g = st.columns(3)
                    with col_r:
                        st.metric("🔴 Red Lights", st.session_state.detection_results['Red'])
                    with col_y:
                        st.metric("🟡 Yellow Lights", st.session_state.detection_results['Yellow'])
                    with col_g:
                        st.metric("🟢 Green Lights", st.session_state.detection_results['Green'])
                    
                    # Display detailed detection information
                    display_detection_info(st.session_state.sample_detections)
            else:
                st.error(f"Could not load sample image: {sample_path}")
        else:
            st.error(f"Sample image file not found: {sample_path}")
    
    # Instructions
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. Select a sample image from the dropdown menu
        2. Click "Detect Traffic Lights" to process the image
        3. View the detection results
        4. Click "Delete Results" to clear the analysis
        
        These sample images demonstrate various traffic light scenarios including:
        - Individual red, yellow and green lights
        - Multiple traffic signals
        - Night-time conditions
        - Challenging detection scenarios
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚦 Traffic Light Detection System | Developed with OpenCV, Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)

# Main app function
def main():
    pass

if __name__ == "__main__":
    main()
