# Photo Colorizer Opencv+Caffe

## ✨ Magical Image Colorizer ✨

Transform your vintage black & white photos into vibrant color masterpieces with the power of Deep Learning!

🚀 **Check out the live version of Photo Colorizer** 🚀
[![Live Application](https://img.shields.io/badge/Live%20Application-Click%20Here-brightgreen)](https://vintagecolorizer.streamlit.app/)

## 📸 Overview

This Streamlit app colorizes black and white images using a pretrained deep learning model. The model is based on the work from Richard Zhang et al. and leverages OpenCV's DNN module to apply learned color mappings.

## 🌈 Features

- 📷 Upload black & white `.jpg`, `.jpeg`, or `.png` images
- ⚙️ Deep Learning model (Caffe) loaded dynamically
- 🎨 Real-time colorization and display
- 💅 Beautiful custom UI with gradient backgrounds and modern styling
- 🔽 Hosted online for instant use

## 🧠 Model Details

- Based on the **Colorful Image Colorization** model by [Richard Zhang](https://richzhang.github.io/colorization/)
- Utilizes a Caffe model with custom layer configuration and learned ab channel color priors

## 🛠️ Installation

1. Clone the repo
```bash
git clone https://github.com/your-username/magical-image-colorizer.git
cd magical-image-colorizer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

## 📁 File Structure

```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── colorize_model/         # Model files (downloaded at runtime)
└── README.md               # This file
```

## 🧾 Requirements

- Python >= 3.8
- Streamlit
- OpenCV
- Pillow
- NumPy

You can install them all using:
```bash
pip install -r requirements.txt
```

## 🌍 Hosting

The app is live and hosted at:  
🔗 **[https://vintagecolorizer.streamlit.app](https://vintagecolorizer.streamlit.app)**

## 🙏 Credits

- [Richard Zhang](https://github.com/richzhang/colorization) for the original colorization model
- [Streamlit](https://streamlit.io/) for the UI framework

---

✨ Created with love using Streamlit and Deep Learning ✨
