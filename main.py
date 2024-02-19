import streamlit as st 
import pandas as pd 
import numpy as np 
import cv2 
from keras.models import load_model
import io
from PIL import Image, ImageOps
import time 
def main():
    # title block
    st.markdown(
        """
        <div style='text-align: center; background-color: #005f69;'>
            <img src='https://www.ueh.edu.vn/images/logo-header.png' alt='Logo UEH'/>
            <h2>University of Economics Ho Chi Minh City – UEH</h2>
            <h3><img src='https://ctd.ueh.edu.vn/wp-content/uploads/2023/07/cropped-TV_trang_CTD.png' alt='Logo CTD UEH' width='100'/> UEH College Of Technology And Design</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    # end title

    st.header("Pneumonia Prediction using CNN predict and Streamlit for frontend")
    
    # đưa về dạng thập phân
    np.set_printoptions(suppress=True)
    
    # xử lý chọn model
    menu = ["Home", "Model TeachableMachine (Recommended)", "Model Xception" ]
    st.sidebar.title('Navigation')
    choice = st.sidebar.selectbox("Choose a model", menu)
    isLoaded = False
    pixels = 0
    
    if choice == "Home":
        st.write("Choose a model in the left Navigatoion first!!!!")
    elif choice == "Model Xception":
        st.subheader("Model CNN using Xception Architecture")
        st.write("Colab code of Xception [here](https://colab.research.google.com/drive/1PAXt85xIxMoWXswwAj7v2VgaGwBNojAx?usp=sharing)")
        with st.spinner("Loading Model..."):
            model = load_model("final_xception_model/best_model.h5", compile=False)
            isLoaded = True
            pixels = 299
    elif choice == "Model TeachableMachine (Recommended)":
        st.subheader("Model using Teachable Machine")
        st.write("Go to homepage of [Teachable Machine](https://teachablemachine.withgoogle.com/)")
        with st.spinner("Loading Model..."):
            model = load_model("teachable_model/keras_model.h5", compile=False)
            isLoaded = True
            pixels = 224
    else: st.write("Choose a model!!!")

    # upload file
    if isLoaded:
        uploaded_image = st.file_uploader("Upload your Xray image")
        data = np.ndarray(shape=(1, pixels, pixels, 3), dtype=np.float32)
        class_names = open("labels.txt", "r").readlines()
        if not (uploaded_image is None):
            img_cap = "File size: " + str(uploaded_image.size) + " kb"
            st.image(uploaded_image, caption=img_cap)
            image_data = uploaded_image.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # resize image 
            size = (pixels, pixels)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            # turn the image into a numpy array
            image_array = np.asarray(image)
            # normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            # image into the array
            data[0] = normalized_image_array
            # predict image
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            st.info("Report: ")
            st.write("Class:", class_name[2:])
            st.write("Confidence Score: ", confidence_score)
            st.info("Result: ")
            if int(class_name[0]) == 1 or (class_name[0] == 0 and confidence_score < 0.8):
                percentx = round(confidence_score,2) * 100
                st.error("The uploaded X-ray film is predicted to have a risk of pneumonia with an accuracy of " + str(percentx) + "%. You should perform specialized tests at the nearest medical facility as soon as possible.")
            else:
                st.success("The model does not predict any abnormalities, but if you have ANY SYMPTOMS of pneumonia, please go to the nearest medical facility immediately.")
    
def footer_h():
    st.warning("These models are trained for educational purposes with limited data (small amount of samples) and have not been verified by experts for accuracy!")
    st.write("GoogleDrive link of Dataset [here](https://drive.google.com/drive/folders/1CdefiFd3O20duRD3koTLVXWmKztdWbx_?usp=sharing)")
    st.write("Dataset on Kaggle [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)")
    st.write("Dataset License [here](https://creativecommons.org/licenses/by/4.0/)")
    
    st.write("Don't have any samples, get xray images below: ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image_sample_xray/normal_1.jpeg", caption="Normal Xray Image 1")
        with open("image_sample_xray/normal_1.jpeg", "rb") as file:
            st.download_button(
                label = "Tải ảnh Normal Xray Image 1",
                data = file,
                file_name = "normal_1.jpeg",
                mime = "image/jpeg"
            )
    with col2:
        st.image("image_sample_xray/pneumonia_1.jpeg", caption="Pneumonia Xray Image 1")
        with open("image_sample_xray/pneumonia_1.jpeg", "rb") as file:
            st.download_button(
                label = "Tải ảnh Pneumonia Xray Image 1",
                data = file,
                file_name = "pneumonia_1.jpeg",
                mime = "image/jpeg"
            )
    with col3:
        st.image("image_sample_xray/pneumonia_5.jpeg", caption="Pneumonia Xray Image 2")
        with open("image_sample_xray/pneumonia_5.jpeg", "rb") as file:
            st.download_button(
                label = "Tải ảnh Pneumonia Xray Image 2",
                data = file,
                file_name = "pneumonia_5.jpeg",
                mime = "image/jpeg"
            )
        
if __name__ == "__main__":
    main()
    
    footer_h()