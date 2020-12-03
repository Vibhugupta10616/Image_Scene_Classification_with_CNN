import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image , ImageOps

st.write(
"""
<font style="color:white">@author: Vibhu Gupta</font>
<br>
<font style="color:white">Date: 3/12/20</font>
"""
,unsafe_allow_html=True)
@st.cache(allow_output_mutation = True)


def load_model():
        CNN_model = tf.keras.models.load_model('CNN_model.hdf5')
        return CNN_model

def model_summary():
        st.sidebar.write("""
                    |**Layer**                   |**Output Shape**     |**Param**|    
                    |----------------------------|---------------------|---------|
                    |conv2d (Conv2D)             | (None, 148, 148, 16)|448|
                    |max_pooling2d (MaxPooling2D)| (None, 74, 74, 16)  |0|
                    |conv2d_1 (Conv2D)           | (None, 72, 72, 32)  |4640|
                    |max_pooling2d_1 (MaxPooling2| (None, 36, 36, 32)  |0|
                    |conv2d_2 (Conv2D)           | (None, 34, 34, 64)  |18496|
                    |max_pooling2d_2 (MaxPooling2| (None, 17, 17, 64)  |0|
                    |conv2d_3 (Conv2D)           | (None, 15, 15, 64)  |36928|
                    |conv2d_4 (Conv2D)           | (None, 13, 13, 64)  |73853|
                    |max_pooling2d_3 (MaxPooling2| (None, 6, 6, 64)    |0|
                    |flatten (Flatten)           | (None, 4608)        |0|
                    |dense (Dense)               | (None, 256)         |1179904|
                    |dense_1 (Dense)             | (None, 6)           |1542|

                  `Total params: 1,315,814`
                  `Trainable params: 1,315,814`
                  `Non-trainable params: 0`
                  
                    """)

        image = Image.open("CNN accuracy graph.png")
        im1 = image.resize((300,250))
        st.sidebar.image(im1)

        st.sidebar.write("""
                             `Training Accuracy:- 98.1%` """)
        
        st.sidebar.write("""
                             `Validation Accuracy:- 80.1%`
                         """)

    


def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)     
        img_reshape = img[np.newaxis,...]   
        prediction = model.predict(img_reshape)
        
        if (0 == prediction.argmax()):
            result = "buildings"
        elif (1 == prediction.argmax()):
            result = "forest"
        elif (2 == prediction.argmax()):
            result = "glacier"
        elif (3 == prediction.argmax()):
            result = "mountain"
        elif (4 == prediction.argmax()):
            result = "sea"
        elif (5 == prediction.argmax()):
            result = "street"

        return result

def main():
    st.set_option('deprecation.showfileUploaderEncoding' , False)
    
    page_bg_img = '''
    <style>	
    body { 
    background-image: url("https://i.ibb.co/7WpmpsH/back-img.jpg");
    background-size: cover;
    }

    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.sidebar.write("## Explore The Model")
    
    model = load_model()
             
    if st.sidebar.button("Simple CNN Model"):
        model_summary()

    st.write(
    """
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:black;text-align:center;">Image scene classification ML App </h2>
    </div>

    <font style="color:white">This is a simple scene image classification web app which predicts whether the image contains <b> Building , Mountain , Forest , Sea , Street</b></font>
    """
        , unsafe_allow_html=True)


    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    
  
    if file is None:
        st.text(" ")
    else:
        img = Image.open(file)
        st.image(img, width=350)
        
        predictions = import_and_predict(img, model)
        string = "This image most likely falls under the category of " + predictions
        st.success(string)    

if __name__ == '__main__':
    main()
