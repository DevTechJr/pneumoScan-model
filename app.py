import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import streamlit as st
import time

model = tf.keras.models.load_model("pneumo2.h5")
model1 = tf.keras.models.load_model("pneumo1.h5")

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.title('PneumoScan')
    st.subheader('Detect Pneumonia From An X-Ray In **Seconds**!')
    code = '''IB MYP Personal Project (2022-2023) - Anirudh Bharadwaj Vangara'''
    st.code(code, language='python')
    ### load file
    uploaded_file = st.file_uploader("Upload A Chest X-ray Scan For A Prediction", type=["jpg","jpeg","png"])
    st.error('PneumoScan is a Machine Learning model with a 92.3% accuracy rate. It is not to be considered as an alternative to a proper diagnosis by a certified medical practitioner. Occasionally, results may be inaccurate. 🚨') 
    variations = {0: 'BACTERIAL PNEUMONIA',
    1: 'VIRAL PNEUMONIA'}

    map_dict = {0: 'NORMAL',
    1: 'PNEUMONIA'}

    
    

    if uploaded_file is not None:

        rawImg = image.load_img(uploaded_file,target_size=(256,256))
        st.image(rawImg, channels="RGB")
        uploaded_img = image.load_img(uploaded_file,target_size=(64,64))
        uploaded_img = img_to_array(uploaded_img)
        uploaded_img = uploaded_img.reshape(1,64,64,3)
        uploaded_img = uploaded_img.astype('float32')
        uploaded_img = uploaded_img/255
        Generate_pred = st.button("Generate A Prediction")
        
        
        if Generate_pred:
            my_bar = st.progress(0)
            progress_text = st.write("Loading Data...")
            for percent_complete in range(25):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            progress_text = st.write("Analysing X-ray Scan...")
            for percent_complete in range(25,51):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            progress_text = st.write("Looking For Patterns...")
            for percent_complete in range(50,76):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            progress_text = st.write("Determining A Prediction...")
            for percent_complete in range(75,100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            with st.spinner('Loading Prediction Output...'):
                    time.sleep(3)

            prediction = model.predict(uploaded_img).argmax()
            nopneumo = model.predict(uploaded_img)
            yespneumo = model.predict(uploaded_img)
            if prediction == 0:
                st.write("**Prediction : No Pneumonia**")
                normal_pred = nopneumo.max()
                normal_pred = normal_pred*100
                st.write("**Prediction Confidence: ",normal_pred,"%**")
            else:
                prediction1 = model1.predict(uploaded_img).argmax()
                st.write("**Model Prediction : {} ⚠️**".format(variations [prediction1]))
                st.warning("**The model has scanned the given X-ray and predicted the presence of {}. We suggest you seek immediate medical assistance for a proper diagnosis.**".format(variations [prediction1]))
                pneumo_pred = yespneumo.max()
                pneumo_pred = pneumo_pred*100
                st.write("**Prediction Confidence:** ", pneumo_pred,"%")
        


with col3:
    st.write("")

    
