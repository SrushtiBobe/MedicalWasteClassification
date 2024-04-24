import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from io import BytesIO

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Load the YOLO model with pre-trained weights
model_path = '/content/gdrive/MyDrive/MedicalWasteClassification/runs/classify/train4/weights/best.pt'
model = YOLO(model_path)

# Define the function to make predictions
def make_predictions(image_data):
    image = Image.open(BytesIO(image_data))
    results = model(image)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    predicted_item_class = names_dict[np.argmax(probs)].split('_')[0]
    predicted_category = names_dict[np.argmax(probs)].split('_')[1]
    return image, predicted_item_class, predicted_category

# Define the function to capture photo and perform waste classification
def take_photo_and_classify():
    picture = st.camera_input("Take a picture")

    if picture:
        # Make prediction
        image, predicted_item_class, predicted_category = make_predictions(picture.getvalue())

        # Display the captured image and prediction
        st.image(image, caption='Captured Image')
        st.write('Predicted Waste Item Class:', predicted_item_class)
        st.write('Predicted Waste Category:', predicted_category)
    else:
        st.warning("No image captured.")

# Define the footer content
def sidebar_footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='text-align: center; font-size: 12px; font-family: Times New Roman; margin-bottom: 5px;'>Medical waste detection and classification</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='text-align: center; font-size: 12px; font-family: Times New Roman;'>Omkar Bhalerao | Srushti Bobe | Priyanka Adhav</div>", unsafe_allow_html=True)

#st.sidebar.markdown("<div style='text-align: center;'><title>Discover</title></div>", unsafe_allow_html=True)
st.sidebar.image("/content/gdrive/MyDrive/MedicalWasteClassification/side-unscreen (1).gif", use_column_width=True)

# Center-align sidebar content and add space after the line
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 30px;'>"
                    "<b>Exploring pathways to innovation and knowledge.</b>"
                    "</div>", unsafe_allow_html=True)

nav_options = ["🏠 Home", "📝 Problem" , "🔧 Working", "📊 Classification", "👩‍💻 About"]
icons = ["➜", "➜", "➜", "➜", "➜"]
for option, icon in zip(nav_options, icons):
    if st.sidebar.button(f"{icon} {option}", key=option):
        st.session_state.page = option.split()[1]
sidebar_footer()

# Render the content based on the selected page
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align: center;'>Medical Waste Detection and Classification</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; font-family:Charmonman'><b>Empowering Health, Protecting Tomorrow: Innovating Medical Waste Detection and Classification</b></p>", unsafe_allow_html=True)
    st.image("/content/gdrive/MyDrive/MedicalWasteClassification/im1 (1).jpg", use_column_width=True)
    st.write("<p style='text-align:justify'>Efficient detection and meticulous classification of medical waste are paramount for safeguarding healthcare professionals, patients, and the environment. Spanning categories such as infectious, sharps, pharmaceutical, and hazardous waste, each demands precise handling and disposal protocols. Accurate identification and categorization empower healthcare facilities to segregate and manage waste streams adeptly, curbing the risk of contamination and ensuring unwavering adherence to regulatory mandates. Robust detection and classification systems not only mitigate potential health and environmental hazards but also exemplify a commitment to responsible medical waste management</p>", unsafe_allow_html=True)

elif st.session_state.page == "Problem":
    st.title("Problem Statement")
    st.write("<p style='text-align:justify'>Inadequate knowledge about medical waste types and management often leads people to dispose of their waste improperly, resulting in mixed garbage that poses challenges for waste collectors during sorting, especially when dealing with hazardous materials.</p>", unsafe_allow_html=True)
    st.image("/content/gdrive/MyDrive/MedicalWasteClassification/prob (1).gif",use_column_width=True)
    st.write("<p style='text-align:justify'>Our solution aims to address this issue by encouraging individuals to segregate their waste according to predefined classifications at the source. Our system facilitates waste classification, providing assistance to individuals who may not be familiar with waste types, thereby promoting proper waste management practices.</p>", unsafe_allow_html=True)


elif st.session_state.page == "Working":
    st.title("System Overview")
    st.write("<p style='text-align:justify'>The outlined process describes a robust system for medical waste detection and classification, ensuring the efficient and safe management of medical waste.</p>", unsafe_allow_html=True)
    st.image("/content/gdrive/MyDrive/MedicalWasteClassification/Pastel (1).gif")
    st.write("<p style='text-align:justify'>The process of medical waste detection and classification initiates with the Input stage, where an image is acquired either through a camera capture or by uploading an existing image file. Once the image is acquired, it proceeds to the Upload phase, where it is transmitted to the system for further analysis. Upon reaching the Detect Image stage, the YOLOv8 object detection algorithm is deployed. YOLOv8 meticulously scrutinizes the image, identifying various objects present within it.</p>", unsafe_allow_html=True)
    st.write("<p style='text-align:justify'>Upon detection, the system classifies these items into four distinct categories: Infectious, Pharmaceutical, Sharps, and Non-Hazardous waste. Specifically:.</p>", unsafe_allow_html=True)

    # List of items
    items = [
        "Gloves are classified as Infectious waste.",
        "Pill Packets fall under the Pharmaceutical category.",
        "Masks are categorized as Infectious waste.",
        "Syringes are identified as Sharps.",
        "Saline Bottles are classified as Non-Hazardous waste.",
        "PPE Kits are categorized as Infectious waste.",
        "Bandages are classified as Infectious waste.",
        "Cardboard is identified as Non-Hazardous waste."
    ]

    # Create an unordered list
    st.write("<ul>", unsafe_allow_html=True)

    # Add list items
    for item in items:
        st.write(f"<li>{item}</li>", unsafe_allow_html=True)

    # Close the unordered list
    st.write("</ul>", unsafe_allow_html=True)

    st.write("<p style='text-align:justify'>This classification enables proper handling, disposal, or treatment based on the hazard level and type of medical waste identified from the input image. By accurately detecting and sorting various types of waste materials using YOLOv8, the system facilitates the efficient and safe management of medical waste, ultimately contributing to improved healthcare waste management practices.</p>", unsafe_allow_html=True)

elif st.session_state.page == "Classification":
    st.markdown("<h1 style='text-align: center;'>Medical Waste Detection and Classification</h1>", unsafe_allow_html=True)
    st.image("/content/gdrive/MyDrive/MedicalWasteClassification/videocon-unscreen (1).gif", use_column_width=True)

    # Streamlit app section for capturing photo and making predictions
    st.markdown("<h4 style='text-align: center;'>Take a Picture for Prediction</h4>", unsafe_allow_html=True)
    st.write(" ➜ Take Picture")
    take_photo_and_classify()


    # Streamlit app section for uploading an image and making predictions
    st.markdown("<h4 style='text-align: center;'>Upload an Image for Prediction</h4>", unsafe_allow_html=True)
    st.write(" ➜ Select Your Image")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file
        image_path = 'uploaded_image.jpg'
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        st.success('File uploaded and saved to {}'.format(image_path))

        # Read the uploaded file as bytes
        image_bytes = uploaded_file.read()

        # Make prediction
        image, predicted_item_class, predicted_category = make_predictions(image_bytes)

        # Display the prediction
        st.subheader('Prediction:')
        st.write('Predicted Waste Item Class:', predicted_item_class)
        st.write('Predicted Waste Category:', predicted_category)

        # Display the uploaded image
        st.subheader('Uploaded Image:')
        st.image(image)




elif st.session_state.page == "About":
    st.title("About Us")

    st.write("<p style='text-align:justify'>We are a team of enthusiastic students from JSPM's Rajarshi Shahu College of Engineering, Pune, pursuing our Bachelor of Technology (B.Tech) in Information Technology. Passionate about technology and innovation, we come together to explore and create solutions that make a positive impact on the world around us.</p>", unsafe_allow_html=True)
    st.write(" ")
    # Set up a single-column layout for the first row
    col1, col2, col3 = st.columns(3)

    # Your photo and name
    with col1:
        st.image("/content/gdrive/MyDrive/MedicalWasteClassification/RBTL21IT010.jpg", use_column_width=True)
        st.write("<p style='text-align:center'>Omkar Bhalerao <br><a href='mailto:omkarbhalerao2002@gmail.com'>omkarbhalerao2002@gmail.com</a></p>", unsafe_allow_html=True)

    # Group member 1 photo and name
    with col2:
        st.image("/content/gdrive/MyDrive/MedicalWasteClassification/Srushti.jpg", use_column_width=True)
        st.write("<p style='text-align:center'>Srushti Bobe<br><a href='mailto:bobesrushti9146@gmail.com'>bobesrushti9146@gmail.com</a></p>", unsafe_allow_html=True)

    # Group member 2 photo and name
    with col3:
        st.image("/content/gdrive/MyDrive/MedicalWasteClassification/Priyanka.jpg", use_column_width=True)
        st.write("<p style='text-align:center'>Priyanka Adhav <br><a href='mailto:adhavpriyanka44@gmail.com'>adhavpriyanka44@gmail.com</a></p>", unsafe_allow_html=True)
        st.write(" ")

    st.write("<p style='text-align:justify'>As students of Rajarshi Shahu College of Engineering, we have access to state-of-the-art facilities and a dynamic learning environment that encourages innovation and collaboration. Our diverse backgrounds and experiences enrich our projects and enable us to approach problems from different perspectives.</p>", unsafe_allow_html=True)

    st.write("Please contact us for more information.")
