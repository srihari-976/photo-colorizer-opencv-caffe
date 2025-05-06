import os
import cv2
import numpy as np
import urllib.request
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import base64
from datetime import datetime

# Custom CSS for a delicious UI
def add_custom_css():
    st.markdown("""
    <style>
        /* Overall page styling */
        .main {
            background-color: #fcfcfc;
            padding: 20px;
        }
        
        /* Custom title styling */
        .title-container {
            background: linear-gradient(90deg, #FF5F6D 0%, #FFC371 100%);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        
        .title-text {
            color: white;
            font-size: 42px;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin: 0;
        }
        
        .subtitle-text {
            color: white;
            font-size: 18px;
            opacity: 0.9;
            margin-top: 10px;
        }
        
        /* File uploader styling */
        .upload-container {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
            margin-bottom: 30px;
            border: 2px dashed #FF5F6D;
        }
        
        /* Results container */
        .results-container {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
            margin-top: 20px;
        }
        
        /* Image display */
        .image-display {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Custom button styling */
        .stButton>button {
            background: linear-gradient(90deg, #FF5F6D 0%, #FFC371 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 24px;
            border-radius: 50px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 95, 109, 0.3);
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #FF5F6D;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #FF5F6D 0%, #FFC371 100%);
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #FF5F6D 0%, #FFC371 100%);
            color: white;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #363636;
            font-weight: 700;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #888888;
            font-size: 14px;
        }
        
        /* Animation for loading */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .loading-animation {
            animation: pulse 1.5s infinite;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to create a delicious UI header
def render_header():
    st.markdown("""
    <div class="title-container">
        <h1 class="title-text">✨ Magical Image Colorizer ✨</h1>
        <p class="subtitle-text">Transform your vintage black & white photos into vibrant color masterpieces</p>
    </div>
    """, unsafe_allow_html=True)

# Function to create gradient text
def gradient_text(text, color1="#FF5F6D", color2="#FFC371"):
    return f"""
    <div style="
        display: inline-block;
        background: -webkit-linear-gradient(left, {color1}, {color2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 24px;
        margin: 10px 0px;
    ">
        {text}
    </div>
    """

def add_footer():
    st.markdown("""
    <div class="footer">
        <p>✨ Created with love using Streamlit and Deep Learning ✨</p>
        <p>© {0} Magical Image Colorizer</p>
    </div>
    """.format(datetime.now().year), unsafe_allow_html=True)

# Function for attractive file uploader
def render_file_uploader():
    st.markdown(gradient_text("Upload Your Image"), unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #666666; margin-bottom: 20px;">
        Choose a black & white photo to transform into a colorful masterpiece!
    </p>
    """, unsafe_allow_html=True)
    
    file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    return file

# Function to create a card container for content
def card_container(title, content):
    st.markdown(f"""
    <div style="
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    ">
        <h3 style="color: #333; border-bottom: 2px solid #FF5F6D; padding-bottom: 10px;">
            {title}
        </h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Add background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Setup page config
st.set_page_config(
    page_title="Magical Image Colorizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS and background
add_custom_css()
add_bg_from_url()

# Render the beautiful header
render_header()

@st.cache_resource
def load_model():
    """Load the colorization model and return it"""
    # Set model paths
    MODEL_DIR = "colorize_model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    PROTOTXT_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt"
    POINTS_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"
    MODEL_URL = "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"

    PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
    POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")
    MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")

    # Download model files
    download_file(PROTOTXT_URL, PROTOTXT)
    download_file(POINTS_URL, POINTS)
    download_file(MODEL_URL, MODEL)

    # Load the model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    return net

def download_file(url, dest):
    """Download a file from a URL if it doesn't exist"""
    if not os.path.exists(dest):
        with st.spinner(f"Downloading {os.path.basename(dest)}..."):
            urllib.request.urlretrieve(url, dest)
    return dest

def colorize_image(image, model):
    """Colorize an image using the pre-trained model"""
    # Convert PIL Image to cv2 format
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Already an RGB image, convert to grayscale for visualization
        image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_bw_3channel = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)
    else:
        # Already grayscale
        image_bw = image
        image_bw_3channel = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)
    
    # Prepare the image for the model
    scaled = image_bw_3channel.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Run the model
    model.setInput(cv2.dnn.blobFromImage(L))
    ab = model.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image_bw.shape[1], image_bw.shape[0]))

    L_original = cv2.split(lab)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return image_bw_3channel, colorized

# Create sidebar with glossy effect
st.sidebar.markdown("""
<div style="
    background: linear-gradient(45deg, rgba(255,95,109,0.8), rgba(255,195,113,0.8));
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(255,95,109,0.3);
">
    <h2 style="text-align: center; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
        ✨ Magic Colorizer
    </h2>
    <p style="text-align: center;">
        Bring your memories to life!
    </p>
</div>
""", unsafe_allow_html=True)

# Add sidebar content
st.sidebar.markdown("""
<div style="
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
">
    <h3 style="color: #FF5F6D; border-bottom: 2px solid #FFC371; padding-bottom: 10px;">
        How It Works
    </h3>
    <p style="color: black">This app uses deep learning to colorize black & white images using a convolutional neural network trained on over a million images.</p>
    <p style="color: black">The AI analyzes patterns and textures to predict the most natural colors for your photo.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
">
    <h3 style="color: #FF5F6D; border-bottom: 2px solid #FFC371; padding-bottom: 10px;">
        Tips for Best Results
    </h3>
    <ul style="color: green">
        <li>Use high-quality black & white images</li>
        <li>Images with clear subjects work best</li>
        <li>Good lighting in the original photo helps</li>
        <li>Historical photos work wonderfully!</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main content area
# Load the model with a custom loading animation
with st.spinner(""):
    st.markdown("""
    <div style="
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
        class="loading-animation"
    ">
        <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/sparkles_2728.png" width="60" class="loading-animation">
        <p>Loading the AI colorization model...</p>
    </div>
    """, unsafe_allow_html=True)
    model = load_model()
    st.success("✨ Magic ready! AI model loaded successfully! ✨")

# File uploader with enhanced UI
uploaded_file = render_file_uploader()

# Process the image if uploaded
if uploaded_file is not None:
    try:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Create a stylish progress container
        st.markdown("""
        <div style="
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
        ">
            <h3 style="color: #FF5F6D;">Processing Your Image</h3>
            <p>Please wait while our AI adds color to your image...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a progress bar for colorization with custom styling
        progress_bar = st.progress(0)
        
        # Simulate progress
        import time
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        
        # Colorize the image
        original, colorized = colorize_image(image, model)
        
        # Complete the progress bar
        progress_bar.progress(100)
        st.success("✨ Colorization complete! ✨")
        
        # Display a beautiful comparison with before/after slider
        st.markdown("""
        <div style="
            background-color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.08);
            text-align: center;
        ">
            <h2 style="
                background: -webkit-linear-gradient(left, #FF5F6D, #FFC371);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 20px;
            ">
                ✨ Before & After Transformation ✨
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display images in columns with fancy borders
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <h3 style="text-align: center; color: #678;">Original Image</h3>
            """, unsafe_allow_html=True)
            st.image(original, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <h3 style="text-align: center; color: #FF5F6D;">Colorized Magic</h3>
            """, unsafe_allow_html=True)
            st.image(colorized, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Option to download the colorized image with beautiful button
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Convert the colorized image to bytes
        colorized_pil = Image.fromarray(colorized)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        colorized_pil.save(temp_file.name)
        
        with open(temp_file.name, "rb") as file:
            st.markdown("""
            <div style="
                text-align: center;
                margin: 30px 0px;
            ">
            """, unsafe_allow_html=True)
            
            btn = st.download_button(
                label="✨ Download Your Colorized Masterpiece ✨",
                data=file,
                file_name=f"colorized_{uploaded_file.name}",
                mime="image/png"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Clean up the temp file
        os.unlink(temp_file.name)
        
        # Add a share section
        st.markdown("""
        <div style="
            background: linear-gradient(90deg, rgba(255,95,109,0.1), rgba(255,195,113,0.1));
            border-radius: 15px;
            padding: 20px;
            margin: 30px 0px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        ">
            <h3 style="color: #FF5F6D;">Share Your Creation</h3>
            <p>Don't forget to share your colorized image with friends and family!</p>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 15px;">
                <button style="
                    background: #3b5998;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    cursor: pointer;
                ">
                    Facebook
                </button>
                <button style="
                    background: #1DA1F2;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    cursor: pointer;
                ">
                    Twitter
                </button>
                <button style="
                    background: #E1306C;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    cursor: pointer;
                ">
                    Instagram
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}")



# FAQ Section
st.markdown("""
<div style="
    background-color: white;
    border-radius: 15px;
    padding: 25px;
    margin: 30px 0px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
">
    <h3 style="
        text-align: center;
        background: -webkit-linear-gradient(left, #FF5F6D, #FFC371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    ">
        Frequently Asked Questions
    </h3>
    <div style="margin-top: 20px;">
        <div style="margin-bottom: 15px;">
            <h4 style="color: #FF5F6D;">How accurate are the colors?</h4>
            <p style="color: green">Our AI model has been trained on millions of images to predict the most natural colors based on patterns and textures. While it can't know the exact original colors, it produces remarkably realistic results!</p>
        </div>
        <div style="margin-bottom: 15px;">
            <h4 style="color: #FF5F6D;">Can I colorize any black & white image?</h4>
            <p style="color: green">Yes! Our tool works with any black & white photograph. Historical photos, family portraits, landscapes - they all work beautifully!</p>
        </div>
        <div style="margin-bottom: 15px;">
            <h4 style="color: #FF5F6D;">Is there a limit to the number of images I can colorize?</h4>
            <p style="color: green">Not at all! Feel free to colorize as many images as you'd like, completely free.</p>
        </div>
        <div style="margin-bottom: 15px;">
            <h4 style="color: #FF5F6D;">How is the quality of the colorized image?</h4>
            <p style="color: green">Our AI produces high-quality colorized images that maintain the original resolution and details of your photo while adding natural-looking colors.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Add a beautiful footer
add_footer()