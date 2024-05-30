import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Tensorflow image predictor function
def predict_image(test_image):

    # Loadinf Model
    model = tf.keras.models.load_model("my_modles.h5")

    # Test Image Preprocessing
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    input_arr = np.array([input_arr])  # Convert single image to batch

    # Prediction
    predictions = model.predict(input_arr)

    return np.argmax(predictions)


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page:",['About Project','Prediction Page'])

if app_mode == 'About Project' :
    st.header('Fruits & Vegetables Recogination System')
    image_path = 'home_img.jpg'
    st.image(image_path,width=4,use_column_width=True)

    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")

    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images in each of the 36 classes)")
    st.text("2. test (10 images in each of the 36 classes)")
    st.text("3. validation (10 images in each of the 36 classes)")

elif app_mode == 'Prediction Page' :
    st.header("Image Prediction")
    test_image = st.file_uploader("Upload Image:")

    if(st.button("Show image")):
        st.image(test_image,width=4,use_column_width=True)

    # Prediction
    if st.button('Predict Image'):
        st.write('Model Prediction:')
        result_index = predict_image(test_image)

        # Reading Lebels
        with open('labels.txt') as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])    

        st.success(label[result_index])    
        





