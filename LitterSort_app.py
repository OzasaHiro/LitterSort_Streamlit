# main_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# クラスのマッピング
CLASSES = ['paper', 'metal', 'cardboard', 'trash', 'glass', 'plastic', 'compost']

@st.cache(allow_output_mutation=True)
def load_model():
    # モデルの読み込み
    interpreter = tf.lite.Interpreter(model_path="littersort_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def classify_image(img, interpreter):
    # 画像のリサイズ
    img = img.resize((224, 224))
    # 画像の正規化
    img_arr = np.array(img).astype(np.float32) / 255.0
    # モデルの入出力の情報を取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 画像を入力データとして設定
    interpreter.set_tensor(input_details[0]['index'], [img_arr])
    # 推論を実行
    interpreter.invoke()
    
    # 推論結果を取得
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_label = np.argmax(output_data)
    
    return CLASSES[pred_label], output_data[0][pred_label]

def main():
    st.title("LitterSortApp")
    uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded_photo', use_column_width=True)
        st.write("")
        st.write("推論中...")
        
        # モデルを読み込む
        interpreter = load_model()
        
        # 画像認識を実行
        label, confidence = classify_image(image, interpreter)
        st.write(f"結果: {label}  (確率: {confidence:.2f}%)")

if __name__ == "__main__":
    main()
