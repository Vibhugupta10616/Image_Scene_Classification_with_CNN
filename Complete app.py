import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image , ImageOps
import base64

st.write(
"""
<font style="color:white">@author: Vibhu Gupta</font>
<br>
<font style="color:white">Date: 3/12/20</font>
"""
,unsafe_allow_html=True)

@st.cache(allow_output_mutation = True)


def select_model(name):
    if name == 'VGG 16 (Transfer Learning)':
        model = tf.keras.models.load_model('VGG16_model.hdf5')
    elif name == 'Simple CNN':
        model = tf.keras.models.load_model('CNN_model.hdf5')
    return model

def model_summary(mod):
    if mod == "VGG 16 (Transfer Learning)":
        st.sidebar.write("""
                    |**Layer (type)**   |**Output Shape**        | **Param**|  
                    |-------------------|------------------------|----------|
                    |input_1            | [(None, 150, 150, 3)]  | 0        | 
                    |block1_conv1       | (None, 150, 150, 64)   | 1792     | 
                    |block1_conv2       | (None, 150, 150, 64)   | 36928    | 
                    |block1_pool        | (None, 75, 75, 64)     | 0        | 
                    |block2_conv1       | (None, 75, 75, 128)    | 73856    | 
                    |block2_conv2       | (None, 75, 75, 128)    | 147584   |
                    |block2_pool        | (None, 37, 37, 128)    | 0        | 
                    |block3_conv        | (None, 37, 37, 256)    | 295168   | 
                    |block3_conv2       | (None, 37, 37, 256)    | 590080   |  
                    |block3_conv3       | (None, 37, 37, 256)    | 590080   | 
                    |block3_pool        | (None, 18, 18, 256)    | 0        | 
                    |block4_conv1       | (None, 18, 18, 512)    | 1180160  | 
                    |block4_conv2       | (None, 18, 18, 512)    | 2359808  | 
                    |block4_conv3       | (None, 18, 18, 512)    | 2359808  | 
                    |block4_pool        | (None, 9, 9, 512)      | 0        | 
                    |flatten            | (None, 41472)          | 0        | 
                    |dense              | (None, 512)            | 21234176 | 
                    |dense_1            | (None, 6)              | 3078     | 

                    `Total params: 28,872,518`
                    `Trainable params: 21,237,254`
                    `Non-trainable params: 7,635,26`
                    """)
        image = Image.open("VGG16 accuracy graph.png")
        im1 = image.resize((300,250))
        st.sidebar.image(im1)

        st.sidebar.write("""
                             `Training Accuracy:- 94.6%` """)
        
        st.sidebar.write("""
                             `Validation Accuracy:- 86.8%`
                         """)

        
        
    elif mod == 'Simple CNN':
        st.sidebar.write("""
                    |**Layer**           |**Output Shape**      | **Param** |    
                    |------------------- |----------------------|-----------|
                    |conv2d              |(None, 148, 148, 16)  | 448       |
                    |max_pooling2d       | (None, 74, 74, 16)   | 0         |
                    |conv2d_1            | (None, 72, 72, 32)   | 4640      |
                    |max_pooling2d_1     | (None, 36, 36, 32)   | 0         |
                    |conv2d_2            | (None, 34, 34, 64)   | 18496     |
                    |max_pooling2d_2     | (None, 17, 17, 64)   | 0         |
                    |conv2d_3            | (None, 15, 15, 64)   | 36928     |
                    |conv2d_4            | (None, 13, 13, 64)   | 73853     |
                    |max_pooling2d_3     | (None, 6, 6, 64)     | 0         |
                    |flatten             | (None, 4608)         | 0         |
                    |dense               | (None, 256)          | 1179904   |
                    |dense_1             | (None, 6)            | 1542      |

                 ` Total params: 1,315,814`
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
    
    main_bg = "back_img.jpg"
    main_bg_ext = "jpg"

    side_bg = "side_img.jpg"
    side_bg_ext = "jpg"

    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
        </style>
        """,
        unsafe_allow_html=True)
        
    st.sidebar.write("## Explore Different Models")
             
    SM = st.sidebar.selectbox('Select Model',('VGG 16 (Transfer Learning)' , 'Simple CNN'))
    model = select_model(SM)
    model_summary(SM)
     
    st.write(
    """
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:black;text-align:center;">Image scene classification ML Web App </h2>
    </div>

     <font style="color:white">This is a simple image scene classification (Deep Learning) web app which predicts whether the image has <b> Buildings , Mountains , Forests , Sea , Street , Galceirs</b></font>
     """
        , unsafe_allow_html=True)
       
    file = st.file_uploader(label = "Please upload an Image", type=["jpg", "png"])
    
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
