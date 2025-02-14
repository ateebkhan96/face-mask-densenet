import streamlit as st
import numpy as np
import tensorflow.lite as tflite
import mediapipe as mp
import cv2
from PIL import Image, ImageOps
import time
from datetime import datetime
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .warning-text {
        color: #dc3545;
        font-weight: bold;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = []
if 'processing_time' not in st.session_state:
    st.session_state['processing_time'] = []

# Load MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection


# Load TFLite Model with error handling
@st.cache_resource
def load_model():
    try:
        interpreter = tflite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def draw_fancy_bbox(image, bbox, label, confidence, confidence_level):
    """
    Draw a fancy bounding box with rounded corners and modern styling
    """
    xmin, ymin, xmax, ymax = bbox
    h, w = image.shape[:2]

    # Calculate dynamic parameters based on image size
    thickness = max(1, int(min(h, w) * 0.003))  # Thinner lines
    corner_radius = max(3, int(min(h, w) * 0.02))  # Dynamic corner radius
    font_scale = min(0.7, max(0.4, min(h, w) * 0.001))  # Smaller font
    padding = max(2, int(min(h, w) * 0.01))  # Dynamic padding

    # Determine color based on prediction and confidence
    if "With Mask" in label:
        if confidence > 0.9:
            color = (0, 200, 0)  # Darker green for high confidence
        else:
            color = (0, 170, 0)  # Lighter green for lower confidence
    else:
        if confidence > 0.9:
            color = (200, 0, 0)  # Darker red for high confidence
        else:
            color = (170, 0, 0)  # Lighter red for lower confidence

    # Draw rounded corners
    def draw_corner(x1, y1, x2, y2, corner_type):
        if corner_type == 'top_left':
            cv2.ellipse(image, (x1 + corner_radius, y1 + corner_radius),
                        (corner_radius, corner_radius), 180, 0, 90, color, thickness)
        elif corner_type == 'top_right':
            cv2.ellipse(image, (x2 - corner_radius, y1 + corner_radius),
                        (corner_radius, corner_radius), 270, 0, 90, color, thickness)
        elif corner_type == 'bottom_left':
            cv2.ellipse(image, (x1 + corner_radius, y2 - corner_radius),
                        (corner_radius, corner_radius), 90, 0, 90, color, thickness)
        elif corner_type == 'bottom_right':
            cv2.ellipse(image, (x2 - corner_radius, y2 - corner_radius),
                        (corner_radius, corner_radius), 0, 0, 90, color, thickness)

    # Draw the main lines
    cv2.line(image, (xmin + corner_radius, ymin), (xmax - corner_radius, ymin), color, thickness)
    cv2.line(image, (xmin + corner_radius, ymax), (xmax - corner_radius, ymax), color, thickness)
    cv2.line(image, (xmin, ymin + corner_radius), (xmin, ymax - corner_radius), color, thickness)
    cv2.line(image, (xmax, ymin + corner_radius), (xmax, ymax - corner_radius), color, thickness)

    # Draw corners
    draw_corner(xmin, ymin, xmax, ymax, 'top_left')
    draw_corner(xmin, ymin, xmax, ymax, 'top_right')
    draw_corner(xmin, ymin, xmax, ymax, 'bottom_left')
    draw_corner(xmin, ymin, xmax, ymax, 'bottom_right')

    # Prepare label with confidence
    label_text = f"{label} ({confidence:.2f})"
    if confidence_level == "Low Confidence":
        label_text += " ⚠️"

    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Draw label background with gradient effect
    gradient_height = text_h + 2 * padding
    for i in range(gradient_height):
        alpha = 1 - (i / gradient_height) * 0.3
        current_color = tuple(int(c * alpha) for c in color)
        cv2.line(image,
                 (xmin, ymin - gradient_height + i),
                 (xmin + text_w + 2 * padding, ymin - gradient_height + i),
                 current_color,
                 1)

    # Draw label text
    cv2.putText(image,
                label_text,
                (xmin + padding, ymin - padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA)

    return image


# Preprocess face image with additional checks
def preprocess_face(face):
    try:
        if face is None or face.size == 0:
            raise ValueError("Invalid face image")

        face = cv2.resize(face, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=0)
        return face
    except Exception as e:
        st.warning(f"Error in preprocessing: {str(e)}")
        return None


# Enhanced prediction function with confidence threshold
def predict(face, interpreter, confidence_threshold=0.7):
    try:
        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        confidence = np.max(output_data)

        if confidence < confidence_threshold:
            return output_data, "Low Confidence"
        return output_data, "High Confidence"
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


# Enhanced detection function
def detect_and_predict(image, interpreter):
    start_time = time.time()
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces_detected = 0

    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1
    ) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                faces_detected += 1
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image_np.shape
                xmin, ymin, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(
                    bbox.height * h)

                # Boundary checking
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmin + width), min(h, ymin + height)

                face_crop = image_np[ymin:ymax, xmin:xmax]

                if face_crop.size > 0:
                    processed_face = preprocess_face(face_crop)
                    if processed_face is not None:
                        prediction, confidence_level = predict(processed_face, interpreter)

                        if prediction is not None:
                            class_labels = ["With Mask", "Without Mask"]
                            predicted_label = class_labels[np.argmax(prediction)]
                            confidence = np.max(prediction)

                            # Draw fancy bounding box
                            image_np = draw_fancy_bbox(
                                image_np,
                                (xmin, ymin, xmax, ymax),
                                predicted_label,
                                confidence,
                                confidence_level
                            )

                            # Store detection results
                            st.session_state['detection_history'].append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'prediction': predicted_label,
                                'confidence': confidence,
                                'confidence_level': confidence_level
                            })

    processing_time = time.time() - start_time
    st.session_state['processing_time'].append(processing_time)

    return Image.fromarray(image_np), faces_detected, processing_time


# Main UI function
def main():
    # Sidebar for settings and statistics
    with st.sidebar:
        st.header("⚙️ Settings & Statistics")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.1)

        if st.session_state['processing_time']:
            st.subheader("📊 Performance Metrics")
            avg_time = np.mean(st.session_state['processing_time'])
            st.metric("Average Processing Time", f"{avg_time:.3f}s")

        if st.session_state['detection_history']:
            st.subheader("🔍 Detection History")
            total_detections = len(st.session_state['detection_history'])
            mask_count = sum(1 for d in st.session_state['detection_history']
                             if d['prediction'] == "With Mask")
            no_mask_count = total_detections - mask_count

            st.write(f"Total Detections: {total_detections}")
            st.write(f"With Mask: {mask_count}")
            st.write(f"Without Mask: {no_mask_count}")

    # Main content
    st.title("😷 Advanced Face Mask Detection System")
    st.markdown("""
        This system uses AI to detect face masks in images and live video feeds.
        Choose your preferred input method below.
    """)

    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Camera Input", "📁 File Upload"])

    with tab1:
        st.subheader("Live Camera Detection")
        cam_quality = st.select_slider("Camera Quality",
                                       options=["Low", "Medium", "High"],
                                       value="Medium")

        if st.button("Start Camera", key="start_camera"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            stop_button = st.button("Stop Camera")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera error. Please check your device.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_image, faces, proc_time = detect_and_predict(
                    Image.fromarray(frame_rgb), interpreter)

                stframe.image(processed_image,
                              caption=f"Detected {faces} faces in {proc_time:.3f}s",
                              use_container_width=True)

                time.sleep(0.1)  # Prevent excessive CPU usage

            cap.release()

    with tab2:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("Upload an image",
                                         type=["jpg", "jpeg", "png"],
                                         help="Supported formats: JPG, JPEG, PNG")

        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                image = image.convert("RGB")
                image = ImageOps.exif_transpose(image)

                # Process image
                processed_image, faces, proc_time = detect_and_predict(image, interpreter)

                # Display results in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(processed_image,
                             caption=f"Detected Image ({faces} faces, {proc_time:.3f}s)",
                             use_container_width=True)

                # Add download button for processed image
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                btn = st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")


# Initialize model and run main function
interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if interpreter:
    main()
else:
    st.error("Failed to load the model. Please check the model file and try again.")