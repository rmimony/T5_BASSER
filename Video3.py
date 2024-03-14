import cv2
import streamlit as st
from ultralytics import YOLO
import keras 
from PIL import Image, ImageOps
import numpy as np
import pandas as pd


# Setting page layout
st.set_page_config(
    page_title="بصير للذوق العام",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Replace the relative path to your weight file



model_path = "best.pt"
Face_model = keras.models.load_model("C:/Users/rmimo/Downloads/FaceModel1.hdf5")
def Face_Detection(image_data,Model):
   
    size = (150,150)
    image_PIL = Image.fromarray(image_data)

    image = ImageOps.fit(image_PIL, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    #prediction = Model.predict([img])
    prediction=Model.predict(img_reshape)
    #st.write(prediction)
    return prediction
    
    
 


# Creating sidebar
with st.sidebar:
    st.markdown(
"""

<div class="avatar">
    <img src="https://i.ibb.co/fYV49ZN/snapedit-1708330789162.png" alt="Bot Avatar">
</div>

""", unsafe_allow_html=True)

    st.header(" ")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    source_vid = st.sidebar.selectbox(
        "السجلات",
        [
         "Videos/24-Feb-2024.mov",
         ])
    
    # Model Options
    confidence =0.5
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
# st.write("Model loaded successfully!")

if source_vid is not None:
    with open(str(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('حدد المخالفة'):
        vid_cap = cv2.VideoCapture(source_vid
            )
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            vid_cap.set(cv2.CAP_PROP_FPS, 9)
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=confidence)
                
                l = res[0]
              
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                #st.write(res)

                #result_tensor1 = face[0].boxes
               # res_plotted1 = face[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
                face=Face_Detection(image,Face_model)              
                #st.write(face)
            #if res[0].probs is not None:
                    #class_indices = res[0].probs.argmax(dim=1)  # Get class indices with highest probability
                   # class_names = [names[i.item()] for i in class_indices]  # Convert indices to class names
                   # st.write(class_names)
                for box in res[0].boxes:
                    class_id = int(box.cls)  # Get class ID
                    class_label = res[0].names[class_id]  # Get class label from class ID
                    print(f'Detected class: {class_label}')  # Print class label
                    #st.write(class_label)
                #for box in res[0].boxes:
                #label =res[0].pred_classes
                #st.write(label)
                #st.write(res[0].boxes.labels)

                st.subheader(':التقرير ')
            
               # ['Throwing_Waste', 'Clothes_Violation', 'Graffiti_infraction','Legal','Garden_Fire']
                if(class_label=='Throwing_Waste'): 
                    df = pd.DataFrame({"command": "st.selectbox", "rating": 4, "is_widget": True},)
                    st.table(df)
                    break
                elif(class_label=='Clothes_Violation'):
                    df = pd.DataFrame([{"رقم الهوية \الإقامة": "1108656745", "الإسم": "عمر بن سلطان بن عبدالله", "المخالفة":".إرتداء الملابس الداخلية وثياب النوم في الأماكن العامة ", "الغرامة":"100 "}])
                    st.table(df)
                    break
                elif(class_label=='Graffiti_infraction'):
                    df = pd.DataFrame([{"رقم الهوية \الإقامة": "1108656745", "الإسم": "عمر بن سلطان بن عبدالله ", "المخالفة":".الكتابة أو الرسم أو مافي حكمهما على وسائل النقل وعلى جدران الأماكن العامة ", "الغرامة":"100 "}])
                    st.table(df)
                    break
                elif(class_label=='Garden_Fire'):
                    df = pd.DataFrame([{"رقم الهوية \الإقامة": "1108656745", "الإسم": "عمر بن سلطان بن عبدالله", "المخالفة":".إشعال النار في الحدائق والأماكن العامة في غير الأماكن المسموح بها ", "الغرامة":"100 "}])
                    st.table(df)
                    break 

               # break
                #Face_Detection(source_vid,Face_model)
                else:
                    vid_cap.release()
                    break