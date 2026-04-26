import streamlit as st
import cv2
import numpy as np

#記錄： Transfer into real website

st.title("Intelligence Image Processing System")
st.write("User Manual: ")
st.write("1 Choose an input method first depending on your own preference")
st.write("2 Follow the steps to continue. ")


input_method = st.radio(
    "Choose Input Method", 
    ["📷Camera", "💾Upload"]
    )


cv2_img = None


if input_method == "📷Camera":
    take_a_photo = st.camera_input("Take a Photo")
    if take_a_photo is not None:
        data = take_a_photo.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(data,np.uint8), cv2.IMREAD_COLOR)
elif input_method == "💾Upload":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


option = st.selectbox(
    'Select a function',
    ('🖼️Original Image', '🔳Grayed out', "😶‍🌫️Blur", "😀Face Detection", "🎨Color Detection")
)

output_img = None

if cv2_img is not None:
    if option == "😀Face Detection" and input_method == "📷Camera":
        st.warning("⚠️Face Detection only supports uploading images")
    else: 
        #Grayed out
        if option == "🔳Grayed out":
            gray = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
            output_img = gray
            st.image(gray)
        
        #Image Blur
        elif option == "😶‍🌫️Blur":
            bgrimg1 = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            blur_level = st.slider("Intensity", 1, 51, 15, step=2)
            blur = cv2.GaussianBlur(bgrimg1, (blur_level,blur_level),0)
            output_img = blur
            st.image(blur)
        
        #Face Detection
        elif option == "😀Face Detection":
            faceCascade = cv2.CascadeClassifier('face_detect.xml')
            gray1 = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray1, 1.5, 1)
            for (x, y, w, h) in faces: #左上座標、長寬度
                cv2.rectangle(cv2_img, (x,y), (x+w, y+h), (0,255,0), 2) #1 要畫得圖片 2左上角的點 3 右下角的點 4 顏色 5 粗度
            output_img = cv2_img
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        
        #Color Detection
        elif option == "🎨Color Detection":
            hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
            col1, col2 = st.columns(2)
            with col1: 
                h_min = st.slider("Hue Min", 0, 179, 0)
                h_max = st.slider("Hue Max", 0,179, 179)
                s_min = st.slider("Sat Min", 0, 255, 0)
                s_max = st.slider("Sat Max", 0, 255, 255)
                v_min = st.slider("Val Min", 0, 255, 0)
                v_max = st.slider("Val Max", 0, 255, 255)

                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, h_max])

                mask = cv2.inRange(hsv,lower, upper)
                result = cv2.bitwise_and(cv2_img, cv2_img, mask=mask)
            with col2: 
                st.image(mask,caption="Mask")
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")
            output_img = result
        elif option == "🖼️Original Image":
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        else:
            gray = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
            output_img = gray
            
if output_img is not None:
    _, buffer = cv2.imencode('.png', output_img)
    st.download_button("Download Image", buffer.tobytes(),"result.png")