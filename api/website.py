import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Детекция ЛЭП", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #f9f9f9;
            color: #2c2c2c;
            font-family: 'Georgia', serif;
        }
        /* Усиление пунктирной обводки uploader'а */
        [data-testid="stFileUploadDropzone"] {
            border: 2px dashed #a0a0a0 !important;
            background-color: #ffffff;
            transition: border 0.3s ease;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border: 2px dashed #555555 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Анализ изображений ЛЭП")

@st.cache_resource
def load_model(model_type):
    if model_type == "Быстрая (YOLO26m)":
        return YOLO("weights/yolo26m.pt") # Путь к легкой модели
    else:
        return YOLO("weights/yolo26m.pt") # Путь к тяжелой (YOLO26x)

col1, col2 = st.columns([1, 2])

with col1:
    model_choice = st.radio("Выбор модели", ("Speed", "Quality"))
    conf_thres = st.slider("Порог уверенности", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    model = load_model(model_choice)

with col2:
    uploaded_file = st.file_uploader("Перетащите фотографию сюда", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner('Обработка...'):
        results = model.predict(source=image, conf=conf_thres)
        res_img = results[0].plot() 
        
    st.image(res_img, caption="Результат инференса", channels="BGR", use_container_width=True)