# main_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pyheif
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from datetime import datetime

# クラスのマッピング
CLASSES = ['cardboard','compost', 'glass', 'metal', 'paper',  'plastic', 'trash']

ANSWERS = ['cardboard','compost', 'glass', 'metal', 'paper',  'plastic', 'trash', 'other', 'unknown'] 

def upload_to_google_drive(image, selected_class):
    """画像をGoogle Driveにアップロードします。"""
    setting_path = 'settings.yaml'
    gauth = GoogleAuth(setting_path)
    gauth.LocalWebserverAuth()  # 認証を行う
    drive = GoogleDrive(gauth)

    filename = f"{selected_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(filename, format='JPEG')
    
    uploaded_file = drive.CreateFile({'title': filename, 'parents': [{'id': '1Sn0z8zKnqi127Qxa2LMnWKX_-o7eAKZw'}]})
    uploaded_file.SetContentFile(filename)
    uploaded_file.Upload()
    
    # オプション: ローカルのファイルを削除
    # os.remove(filename)

def open_image(file):
    """ファイルの拡張子に応じて画像を開きます。"""
    try:
        # HEIC形式の場合
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

# クラス名に基づくコメントの辞書
CLASS_COMMENTS = {
    'paper': "This is likely paper. Please recycle it according to local rules.",
    'metal': "This is probably metal. Please recycle it properly.",
    'cardboard': "Looks like cardboard. Recycle it at appropriate facilities.",
    'trash': "This seems to be general trash. Dispose of it correctly.",
    'glass': "This appears to be glass. Handle with care and recycle.",
    'plastic': "This is probably plastic. Recycle if possible.",
    'compost': "This seems to be compostable material. Dispose in a green bin."
}

# 追加のクラス名に基づくコメントの辞書
ADDITIONAL_CLASS_COMMENTS = {
    'others': "This category is for other items not listed.",
    'unknown': "The content of the image is unclear or uncertain."
}

def generate_comment(class_name):
    """指定されたクラス名に基づいてコメントを生成します。"""
    return CLASS_COMMENTS.get(class_name, "Please dispose of or recycle according to local rules.")



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
    uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg', 'heic'])

    #if uploaded_file:
        # 新しい画像がアップロードされた場合、セッションステートをリセット
    #    st.session_state.class_selection = None
    #    st.session_state.upload_clicked = False

    if uploaded_file is not None:
        image = open_image(uploaded_file)
        if image:
            image = image.resize((image.width // 2, image.height // 2))
            st.image(image, caption='Uploaded_photo', use_column_width=True)
            st.write("")
            #st.write("Inference...")
        
            # モデルを読み込む
            interpreter = load_model()
            
            # 画像認識を実行
            label, confidence = classify_image(image, interpreter)
            st.write(f"Result: {label}  (Confidence: {100*confidence:.0f}%)")
            st.write(generate_comment(label))
            

            # クラス選択の状態をセッションステートに保存
            if 'class_selection' not in st.session_state:
                st.session_state.class_selection = None
            
            # Uploadボタンの状態をセッションステートに保存
            if 'upload_clicked' not in st.session_state:
                st.session_state.upload_clicked = False
            
            if st.button('Start Upload'):
                st.session_state.upload_clicked = True
            
            # Uploadボタンがクリックされた場合のみ選択ボックスを表示
            if st.session_state.upload_clicked:
                st.session_state.class_selection = st.selectbox(
                    "What is this photo of?",
                    list(CLASS_COMMENTS.keys()) + list(ADDITIONAL_CLASS_COMMENTS.keys()),
                    index=0 if st.session_state.class_selection is None else list(CLASS_COMMENTS.keys()).index(st.session_state.class_selection)
                )
            
                if st.button("Confirm and Upload to Google Drive"):
                    upload_to_google_drive(image, st.session_state.class_selection)
                    st.write(f"Uploaded {st.session_state.class_selection} image to Google Drive!")



if __name__ == "__main__":
    main()
