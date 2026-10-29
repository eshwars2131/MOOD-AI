import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MOOD.AI | Expression Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# FUTURISTIC UI
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #050816, #0b1630, #101b35);
    color: #e8f1ff;
}

[data-testid="stSidebar"] {
    background: #080e20;
    border-right: 1px solid #193b70;
}

h1, h2, h3 {
    color: #76b7ff !important;
}

p, label {
    color: #e8f1ff;
}

div.stButton > button {
    background: linear-gradient(90deg, #1769ff, #6c42ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

div.stButton > button:hover {
    border: 1px solid #76b7ff;
    color: white;
}

[data-testid="stMetric"] {
    background: #101e38;
    border: 1px solid #244775;
    padding: 18px;
    border-radius: 12px;
}

hr {
    border-color: #244775;
}

div[data-testid="stProgressBar"] > div > div > div {
    background-color: #4c91ff;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_photo_id" not in st.session_state:
    st.session_state.last_photo_id = None

# =====================================================
# HELPER: LOAD OPENCV FACE DETECTOR
# =====================================================

@st.cache_resource
def load_face_detector():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "OpenCV could not load the Haar Cascade XML file."
        )

    return detector


# =====================================================
# HELPER: DETECT AND CROP FACE
# =====================================================

def detect_and_crop_face(image, detector, min_face_size=70):
    """
    Detect faces with OpenCV and return the largest face crop.

    The returned crop is in BGR format, which is the format
    expected by DeepFace when supplied with a NumPy image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast slightly for face detection.
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size)
    )

    if len(faces) == 0:
        return None, None

    # Choose the largest detected face.
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    image_height, image_width = image.shape[:2]

    # Add padding around the face so the crop includes
    # the forehead, cheeks, and chin.
    padding_x = int(w * 0.20)
    padding_y = int(h * 0.25)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image_width, x + w + padding_x)
    y2 = min(image_height, y + h + padding_y)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None, None

    coordinates = (x, y, w, h)

    return face_crop, coordinates


# =====================================================
# HELPER: NORMALIZE DEEPFACE OUTPUT
# =====================================================

def normalize_analysis_result(result):
    if isinstance(result, list):
        if not result:
            raise ValueError(
                "DeepFace returned an empty result."
            )

        result = result[0]

    if not isinstance(result, dict):
        raise ValueError(
            "DeepFace returned an unexpected result format."
        )

    scores = result.get("emotion", {})

    if not scores:
        raise ValueError(
            "DeepFace did not return expression scores."
        )

    clean_scores = {
        str(name): float(value)
        for name, value in scores.items()
    }

    return clean_scores


# =====================================================
# HEADER
# =====================================================

st.title("🎭 MOOD.AI")
st.markdown("### AI-Powered Facial Expression Recognition")

st.write(
    "A computer-vision project combining OpenCV face detection "
    "with a pretrained DeepFace expression classifier."
)

st.info(
    "🔒 Privacy: Images are processed by this Python application. "
    "This app does not intentionally save captured images to its "
    "history. Facial-expression predictions are estimates, not "
    "proof of a person's actual feelings."
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 🧠 MOOD.AI")
    st.caption("AI Expression Analyzer")

    st.divider()

    st.markdown("### ⚙️ Settings")

    min_face_size = st.slider(
        "Minimum face size",
        min_value=40,
        max_value=180,
        value=70,
        step=10,
        help=(
            "Lower this if your face is small in the camera image. "
            "Higher values may reduce false detections."
        )
    )

    st.markdown("### 🎯 Features")
    st.write("📸 Webcam capture")
    st.write("🔍 OpenCV face detection")
    st.write("🧠 DeepFace expression analysis")
    st.write("📊 Expression score chart")
    st.write("🕘 Session history")
    st.write("⬇️ CSV export")

    st.divider()

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):
        st.session_state.history = []
        st.session_state.last_photo_id = None
        st.rerun()

    st.caption("AI Subject Assignment")


# =====================================================
# CAMERA AND RESULTS
# =====================================================

camera_col, result_col = st.columns(
    [1.1, 1],
    gap="large"
)

with camera_col:
    st.subheader("📸 Camera Studio")

    photo = st.camera_input(
        "Capture your face",
        help=(
            "Face the camera directly. Use even lighting and "
            "avoid a bright window directly behind you."
        )
    )

with result_col:
    st.subheader("🔍 AI Analysis")

    if photo is None:
        st.markdown("""
        <div style="
            background:#101e38;
            padding:30px;
            border-radius:15px;
            border:1px solid #244775;
            text-align:center;">
            <h2>🤖 Ready to Analyze</h2>
            <p>Capture a photo to begin.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        try:
            # ---------------------------------------------
            # DECODE IMAGE
            # ---------------------------------------------

            image_bytes = np.frombuffer(
                photo.getvalue(),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                image_bytes,
                cv2.IMREAD_COLOR
            )

            if image is None:
                st.error(
                    "Unable to decode the captured image."
                )

            else:
                image_height, image_width = image.shape[:2]

                st.caption(
                    f"Image dimensions: {image_width} × "
                    f"{image_height} pixels"
                )

                # -----------------------------------------
                # FACE DETECTION
                # -----------------------------------------

                with st.spinner(
                    "🔍 Detecting face with OpenCV..."
                ):
                    face_detector = load_face_detector()

                    face_crop, face_box = detect_and_crop_face(
                        image,
                        face_detector,
                        min_face_size=min_face_size
                    )

                if face_crop is None:
                    st.error(
                        "No face was detected. Please capture "
                        "another photo with your face closer to "
                        "the camera and clearly lit."
                    )

                    st.markdown("**Try these tips:**")
                    st.write("- Face the camera directly.")
                    st.write("- Move away from bright windows.")
                    st.write("- Remove anything covering your face.")
                    st.write(
                        "- Lower the minimum face size in the sidebar "
                        "if your face appears small."
                    )

                    st.image(
                        cv2.cvtColor(
                            image,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Captured image — no face detected",
                        width="stretch"
                    )

                else:
                    x, y, w, h = face_box

                    # -------------------------------------
                    # SHOW DETECTED FACE
                    # -------------------------------------

                    st.success(
                        f"Face detected! Bounding box: "
                        f"{w} × {h} pixels."
                    )

                    preview_col1, preview_col2 = st.columns(2)

                    with preview_col1:
                        preview_image = image.copy()

                        cv2.rectangle(
                            preview_image,
                            (x, y),
                            (x + w, y + h),
                            (60, 180, 255),
                            3
                        )

                        st.image(
                            cv2.cvtColor(
                                preview_image,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Detected face",
                            width="stretch"
                        )

                    with preview_col2:
                        st.image(
                            cv2.cvtColor(
                                face_crop,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Face crop used for AI analysis",
                            width="stretch"
                        )

                    # -------------------------------------
                    # DEEPFACE ANALYSIS
                    # -------------------------------------

                    with st.spinner(
                        "🧠 Analyzing facial expression..."
                    ):
                        # The face has already been detected
                        # and cropped. The 'skip' backend avoids
                        # attempting a second face detection.
                        result = DeepFace.analyze(
                            img_path=face_crop,
                            actions=["emotion"],
                            enforce_detection=False,
                            detector_backend="skip",
                            silent=True
                        )

                    scores = normalize_analysis_result(result)

                    # -------------------------------------
                    # PREDICTION
                    # -------------------------------------

                    predicted = max(
                        scores,
                        key=scores.get
                    )

                    score = scores[predicted]

                    st.success("Expression analysis completed!")

                    metric1, metric2 = st.columns(2)

                    with metric1:
                        st.metric(
                            "Top Expression Label",
                            predicted.upper()
                        )

                    with metric2:
                        st.metric(
                            "Top Model Score",
                            f"{score:.1f}%"
                        )

                    st.caption(
                        "The top model score is not a calibrated "
                        "probability that this person feels the "
                        "predicted emotion. Expression classifiers "
                        "can make mistakes."
                    )

                    # -------------------------------------
                    # SCORES
                    # -------------------------------------

                    st.markdown("### 📊 Expression Scores")

                    sorted_scores = sorted(
                        scores.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )

                    for expression, value in sorted_scores:
                        st.write(
                            f"**{expression.capitalize()}** — "
                            f"{value:.1f}%"
                        )

                        st.progress(
                            min(
                                1.0,
                                max(0.0, value / 100.0)
                            )
                        )

                    # -------------------------------------
                    # CHART
                    # -------------------------------------

                    st.markdown("### 📈 Score Visualization")

                    chart_data = pd.DataFrame({
                        "Expression": list(scores.keys()),
                        "Score (%)": list(scores.values())
                    })

                    st.bar_chart(
                        chart_data.set_index("Expression")
                    )

                    # -------------------------------------
                    # HISTORY
                    # -------------------------------------

                    photo_id = photo.getvalue()

                    if (
                        st.session_state.last_photo_id
                        != photo_id
                    ):
                        record = {
                            "Time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "Expression": predicted.capitalize(),
                            "Model Score (%)": round(score, 2),
                            "Face Width (px)": int(w),
                            "Face Height (px)": int(h)
                        }

                        st.session_state.history.append(record)

                        st.session_state.last_photo_id = photo_id

        except Exception as error:
            st.error(
                "Analysis failed. Please try another photo. "
                "If the problem continues, inspect the technical "
                "details below."
            )

            with st.expander("Technical error details"):
                st.code(str(error))


# =====================================================
# SESSION HISTORY
# =====================================================

st.divider()

st.subheader("🕘 Session History")

if st.session_state.history:
    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv_data,
        file_name="mood_ai_history.csv",
        mime="text/csv"
    )

else:
    st.caption(
        "Your analysis history will appear here after a prediction."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:10px;">
    <p>MOOD.AI | Artificial Intelligence Project</p>
    <p>Computer Vision • Deep Learning • Image Classification</p>
</div>
""", unsafe_allow_html=True)