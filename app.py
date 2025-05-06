import os
import numpy as np
import urllib.request
import streamlit as st
from datetime import datetime
from PIL import Image
import streamlit.components.v1 as components

# Try importing OpenCV - if not available, inform the user
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# -------------------- UI Customization --------------------
def add_custom_css():
    st.markdown("""
    <style>
        .main { background-color: #fcfcfc; padding: 20px; }
        .title-container {
            background: linear-gradient(90deg, #FF5F6D 0%, #FFC371 100%);
            padding: 20px; border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1); margin-bottom: 30px; text-align: center;
        }
        .title-text {
            color: white; font-size: 42px; font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2); margin: 0;
        }
        .subtitle-text {
            color: white; font-size: 18px; opacity: 0.9; margin-top: 10px;
        }
        .upload-container {
            background-color: #ffffff; border-radius: 15px; padding: 30px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.05); margin-bottom: 30px;
            border: 2px dashed #FF5F6D;
        }
        .results-container {
            background-color: #ffffff; border-radius: 15px; padding: 20px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.05); margin-top: 20px;
        }
        .image-display {
            border-radius: 12px; overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF5F6D 0%, #FFC371 100%);
            color: white; font-weight: 600; border: none;
            padding: 12px 24px; border-radius: 50px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 95, 109, 0.3);
        }
        .stProgress > div > div { background-color: #FF5F6D; }
        .css-1d391kg, .sidebar .sidebar-content {
            background: linear-gradient(180deg, #FF5F6D 0%, #FFC371 100%);
            color: white;
        }
        h1, h2, h3 {
            color: #363636;
            font-weight: 700;
        }
        .footer {
            text-align: center; margin-top: 50px; padding: 20px;
            color: #888888; font-size: 14px;
        }
        .error-container {
            background-color: #fff3f3; 
            border-left: 5px solid #FF5F6D; 
            padding: 20px; 
            margin: 20px 0;
            border-radius: 5px;
        }
        .install-instructions {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div class="title-container">
        <h1 class="title-text">✨ Magical Image Colorizer ✨</h1>
        <p class="subtitle-text">Transform your vintage black & white photos into vibrant color masterpieces</p>
    </div>
    """, unsafe_allow_html=True)

def gradient_text(text, color1="#FF5F6D", color2="#FFC371"):
    return f"""
    <div style="display:inline-block;
        background: -webkit-linear-gradient(left, {color1}, {color2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold; font-size: 24px; margin: 10px 0px;">
        {text}
    </div>
    """

def add_footer():
    st.markdown(f"""
    <div class="footer">
        <p>✨ Created with love using Streamlit and Deep Learning ✨</p>
        <p>© {datetime.now().year} Magical Image Colorizer</p>
    </div>
    """, unsafe_allow_html=True)

def render_file_uploader():
    st.markdown(gradient_text("Upload Your Image"), unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #666666; margin-bottom: 20px;">
        Choose a black & white photo to transform into a colorful masterpiece!
    </p>
    """, unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    return file

def add_bg_from_url():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Model Loading --------------------
@st.cache_resource
def load_model():
    if not CV2_AVAILABLE:
        return None
        
    MODEL_DIR = "colorize_model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    PROTOTXT_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt"
    POINTS_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"
    MODEL_URL = "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"

    PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
    POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")
    MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")

    download_file(PROTOTXT_URL, PROTOTXT)
    download_file(POINTS_URL, POINTS)
    download_file(MODEL_URL, MODEL)

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS).transpose().reshape(2, 313, 1, 1)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

def download_file(url, dest):
    if not os.path.exists(dest):
        with st.spinner(f"Downloading {os.path.basename(dest)}..."):
            urllib.request.urlretrieve(url, dest)

# -------------------- Image Colorization --------------------
def colorize_image(image, model):
    if not CV2_AVAILABLE or model is None:
        return None, None
        
    image = np.array(image.convert("RGB"))
    image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_bw_3channel = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)

    scaled = image_bw_3channel.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    model.setInput(cv2.dnn.blobFromImage(L))
    ab = model.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image_bw.shape[1], image_bw.shape[0]))

    L_original = cv2.split(lab)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return image_bw_3channel, colorized

# -------------------- Streamlit Page Setup --------------------
def main():
    st.set_page_config(
        page_title="Magical Image Colorizer",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    add_custom_css()
    add_bg_from_url()
    render_header()

    # Sidebar Info
    st.sidebar.markdown("""
    <div style="background: linear-gradient(45deg, rgba(255,95,109,0.8), rgba(255,195,113,0.8));
    padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(255,95,109,0.3);">
        <h2 style="text-align: center;">✨ Magic Colorizer</h2>
        <p style="text-align: center;">Bring your memories to life!</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h3 style="color: #FF5F6D;">How It Works</h3>
        <p style="color: green">This app uses deep learning to colorize black & white images using a CNN trained on over a million photos.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if OpenCV is available
    if not CV2_AVAILABLE:
        st.markdown("""
        <div class="error-container">
            <h3>⚠️ OpenCV Not Found</h3>
            <p>The OpenCV (cv2) library is required to run this application but it's not installed in your environment.</p>
            <p>To fix this issue, install OpenCV by adding it to your requirements.txt file or running:</p>
            <div class="install-instructions">pip install opencv-python-headless</div>
            <p>For Streamlit Cloud deployment, make sure to include this in your requirements.txt file:</p>
            <div class="install-instructions">opencv-python-headless==4.5.5.64</div>
            <p>Note: The headless version is recommended for cloud deployments.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display example images
        st.markdown(gradient_text("Example Colorized Images"), unsafe_allow_html=True)
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            st.image("https://i.imgur.com/lZVPBn8.jpg", caption="Original B&W Example", use_column_width=True)
        with example_col2:
            st.image("https://i.imgur.com/RxnF3AM.jpg", caption="Colorized Example", use_column_width=True)
            
    else:
        # Upload and process image
        file = render_file_uploader()
        model = load_model()

        if file:
            image = Image.open(file)
            st.markdown("### Original and Colorized Results")
            with st.spinner("Colorizing your image... ✨"):
                bw_img, colorized_img = colorize_image(image, model)

            col1, col2 = st.columns(2)
            with col1:
                st.image(bw_img, caption="Original B&W", use_column_width=True)
            with col2:
                st.image(colorized_img, caption="Colorized", use_column_width=True)

    add_footer()

if __name__ == "__main__":
    main()
