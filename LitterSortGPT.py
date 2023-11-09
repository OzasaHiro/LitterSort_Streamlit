# main_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pyheif
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from datetime import datetime
from rembg import remove
from openai import OpenAI
import os
import base64
import streamlit as st
from io import BytesIO

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

#class mapping
CLASSES = ['cardboard','compost', 'glass', 'metal', 'paper',  'plastic', 'trash']

ANSWERS = ['cardboard','compost', 'glass', 'metal', 'paper',  'plastic', 'trash', 'other', 'unknown'] 

def upload_to_google_drive(image, class_name, selected_class):
    """upload to GoogleDrive"""
    setting_path = 'settings.yaml'
    gauth = GoogleAuth(setting_path)
    gauth.LocalWebserverAuth()  
    drive = GoogleDrive(gauth)

    i = 1 if class_name == selected_class else 0

    filename = f"{selected_class}_{i}_{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(filename, format='JPEG')
    
    uploaded_file = drive.CreateFile({'title': filename, 'parents': [{'id': '1Sn0z8zKnqi127Qxa2LMnWKX_-o7eAKZw'}]})
    uploaded_file.SetContentFile(filename)
    uploaded_file.Upload()
    
    #Option delete the local file
    # os.remove(filename)

def open_image(file):
    """open image file"""
    try:
        # for HEIC type
        if file.name.lower().endswith(".heic"):
            heif_file = pyheif.read(file.getvalue())
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        else:
            image = Image.open(file)
    except Exception as e:
        st.error(f"Loading Error: {e}")
        return None

    return image

# dictionaly of class and comments
CLASS_COMMENTS = {
    'paper': "This is likely paper. Please recycle it according to local rules.",
    'metal': "This is probably metal. Please recycle it properly.",
    'cardboard': "Looks like cardboard. Recycle it at appropriate facilities.",
    'trash': "This seems to be general trash. Dispose of it correctly.",
    'glass': "This appears to be glass. Handle with care and recycle.",
    'plastic': "This is probably plastic. Recycle if possible.",
    'compost': "This seems to be compostable material. Dispose in a green bin."
}

# additional class
ADDITIONAL_CLASS_COMMENTS = {
    'others': "This category is for other items not listed.",
    'unknown': "The content of the image is unclear or uncertain."
}

def generate_comment(class_name):
    """create comments"""
    return CLASS_COMMENTS.get(class_name, "Please dispose of or recycle according to local rules.")


@st.cache(allow_output_mutation=True)
def load_model():
    # load ML model
    interpreter = tf.lite.Interpreter(model_path="littersort_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def classify_image(img, interpreter):
    # image resize
    img = img.resize((224, 224))
    # normalization
    img_arr = np.array(img).astype(np.float32) / 255.0
    # input output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    
    interpreter.set_tensor(input_details[0]['index'], [img_arr])
    
    interpreter.invoke()
    
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_label = np.argmax(output_data)
    
    return CLASSES[pred_label], output_data[0][pred_label]

# Function to encode the image
#def encode_image(image_path):
#    with open(image_path, "rb") as image_file:
#        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image(image):
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        #image.save(buffered, format="JPEG")  # or "PNG", depending on your image format
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    else:
        raise TypeError("The function requires a PIL.Image.Image object")


def main():
    st.title("LitterSortApp")
    uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg', 'heic'])

    if uploaded_file is not None:
        image = open_image(uploaded_file)
        if image:
            image = image.resize((image.width // 2, image.height // 2))
            image_bg = image
            image = remove(image)
            img_rgb = image.convert('RGB')

            if image.mode == 'RGBA':
                r, g, b, a = image.split()
                bg_white = Image.new('RGB', image.size, (255, 255, 255))
                bg_white.paste(img_rgb, mask=a)
                image = bg_white
                
            st.image(image, caption='Uploaded_photo', use_column_width=True)
            st.write("")

            # Getting the base64 string
            base64_image = encode_image(image)

            #st.write("We got base64 format!")
            
            interpreter = load_model()

            label, confidence = classify_image(image, interpreter)
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "type": "Please advise on how to dispose of glass as waste in San Jose."},
                            #{
                            #    "type": "image_url",
                            #    "image_url": {
                            #        "url": f"https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg" },
                            #},
                        ],
                    }
                ],
                max_tokens=100,
            )

            st.write(response.choices[0].message.content)           
        
            interpreter = load_model()

            label, confidence = classify_image(image, interpreter)
            st.write(f"Result: {label}  (Confidence: {100*confidence:.0f}%)")
            st.write(generate_comment(label))
            
            if 'class_selection' not in st.session_state:
                st.session_state.class_selection = None
            
            if st.session_state.class_selection in CLASS_COMMENTS.keys():
                index = list(CLASS_COMMENTS.keys()).index(st.session_state.class_selection)
            else:
                index = 0

            st.session_state.class_selection = st.selectbox(
                "What is this photo of? Please let me know the answer!!",
                list(CLASS_COMMENTS.keys()) + list(ADDITIONAL_CLASS_COMMENTS.keys()),
                index=index
            )
            
            if st.session_state.class_selection in CLASS_COMMENTS.keys() or st.session_state.class_selection in ADDITIONAL_CLASS_COMMENTS.keys():
                predicted_label = st.session_state.class_selection
            else:
                predicted_label = None

            if st.button("Upload image"):
                if predicted_label is None:
                    st.error("Please select a class before uploading.")
                else:
                    upload_to_google_drive(image_bg, label, predicted_label)
                    st.success("Uploaded successfully!")
        



if __name__ == "__main__":
    main()
