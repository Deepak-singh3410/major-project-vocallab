import streamlit as st
import numpy as np
import tempfile
import base64
from fpdf import FPDF

# SAFE IMPORT FOR MIC
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    MIC_AVAILABLE = True
except:
    MIC_AVAILABLE = False

from hierarchical_inference import predict

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Vocal Diagnosis", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {background-color: #0e1117;}
h1, h2, h3 {color: #00ffcc;}

.nav-btn button {
    width: 100%;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FAST NAVIGATION ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

nav1, nav2, nav3 = st.columns(3)

with nav1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"

with nav2:
    if st.button("🧪 Diagnosis"):
        st.session_state.page = "Diagnosis"

with nav3:
    if st.button("ℹ️ About"):
        st.session_state.page = "About"

st.markdown("---")

# ---------------- AUDIO RECORD ----------------
def record_audio(duration=5, fs=44100):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, audio)
    return temp_file.name

# ---------------- PDF REPORT ----------------
def generate_pdf(name, age, result, confidence):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Vocal Diagnosis Report", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age}", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Diagnosis Result", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Condition: {result}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}", ln=True)

    pdf.ln(10)

    if result != "Normal":
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, "Medical attention recommended", ln=True)
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(200, 10, "No abnormalities detected", ln=True)

    return pdf.output(dest="S").encode("latin-1")

# ---------------- HOME ----------------
def home():
    st.title("🎤 Vocal Disease Detection System Made by Deepak singh and Deepka sharma ")
    st.write("""
    Detect:
    - Laryngozele
    - Vox Senilis
    - Parkinson’s
    - Normal
    """)

# ---------------- DIAGNOSIS ----------------
def diagnosis():
    st.title("🧪 Voice Diagnosis")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.text_input("Age")

    with col2:
        options = ["Upload"]
        if MIC_AVAILABLE:
            options.append("Record")

        option = st.radio("Input Method", options)

    audio_path = None

    # -------- Upload --------
    if option == "Upload":
        file = st.file_uploader("Upload WAV file", type=["wav"])
        if file:
            st.audio(file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                audio_path = tmp.name

    # -------- Record --------
    elif option == "Record":
        if st.button("🎙 Record"):
            audio_path = record_audio()
            st.audio(audio_path)

    # -------- Prediction --------
    if audio_path:
        st.info("Processing...")

        result = predict(audio_path)

        if isinstance(result, tuple):
            label = result[0]
            confidence = float(result[1]) if len(result) > 1 else 0.5
        else:
            label = result
            confidence = 0.5

        st.success(f"Diagnosis: {label}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Condition", label)
        c2.metric("Confidence", f"{confidence:.2f}")
        c3.metric("Status", "Healthy" if label == "Normal" else "Abnormal")

        st.progress(confidence)

        if label != "Normal":
            st.error("⚠️ Possible pathology detected")
        else:
            st.success("✅ Healthy")

        if name and age:
            pdf = generate_pdf(name, age, label, confidence)

            st.download_button(
                "📄 Download Report",
                data=pdf,
                file_name="report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Enter patient details to download report")

# ---------------- ABOUT ----------------
def about():
    st.title("About")
    st.write("AI system for vocal disease detection.")

# ---------------- ROUTER ----------------
if st.session_state.page == "Home":
    home()
elif st.session_state.page == "Diagnosis":
    diagnosis()
elif st.session_state.page == "About":
    about()