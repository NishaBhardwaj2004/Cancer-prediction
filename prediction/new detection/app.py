import pandas as pd
import numpy as np
import streamlit as st
import pickle
import base64

# Load the entire pipeline, not just the model
with open("/workspaces/Cancer-prediction/prediction/new detection/model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

with open("/workspaces/Cancer-prediction/prediction/new detection/model2.pkl", "rb") as f2:
    model2_pipeline = pickle.load(f2)

# Your dictionary of attributes
dict_att = {
    "radius_mean": None, "texture_mean": None, "perimeter_mean": None, "area_mean": None, 
    "smoothness_mean": None, "compactness_mean": None, "concavity_mean": None, 
    "concave points_mean": None, "symmetry_mean": None, "fractal_dimension_mean": None, 
    "radius_se": None, "texture_se": None, "perimeter_se": None, "area_se": None, 
    "smoothness_se": None, "compactness_se": None, "concavity_se": None, 
    "concave_points_se": None, "symmetry_se": None, "fractal_dimension_se": None, 
    "radius_worst": None, "texture_worst": None, "perimeter_worst": None, "area_worst": None, 
    "smoothness_worst": None, "compactness_worst": None, "concavity_worst": None, 
    "concave points_worst": None, "symmetry_worst": None, "fractal_dimension_worst": None
}

# Page configuration
def page_config():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="🧊",
        layout="wide"
    )

# Function to set background image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        width: cover;
        height: cover;
        border: 2px solid;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to display image
def image():
    st.image("/workspaces/Cancer-prediction/prediction/new detection/pic2.png")

# Main function
def main():
    submit = 0
    flag = 0
    st.markdown("<h1 style='text-align:center; color:black;'>Breast Cancer Predictor</h1>", unsafe_allow_html=True)
    st.write("   ")
    st.write("<h4 style='color:black;'>Please connect this app with your cytology lab to help diagnose breast cancer from your sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on measurements it receives from your cytosis lab. Enter the details of parameters you have.</h4>", unsafe_allow_html=True)
    st.write("<h3 style='color:grey'>Select parameters you want to enter from below:</h3>", unsafe_allow_html=True)

    # Select features to enter
    l1 = st.multiselect("", list(dict_att.keys()))
    st.write("\n\n\n\n")
    st.sidebar.title("Enter the details of cell here:-")
    for i in l1:
        x = st.sidebar.number_input(f"Enter the value of {i}:")
        dict_att[i] = x

    # Button for submission
    button1 = st.sidebar.button("Submit")
    if button1:
        submit = 1

    with st.container() as co:
        # Convert inputs to list
        test = list(dict_att.values())
        test = test[0:29]
        ta = np.array(test).reshape(1, -1)

        # Check if all inputs are provided
        for i in test:
            if i is None:
                flag += 1

        if submit == 1:
            if flag == 0:
                # All inputs are provided, use model2
                predict = model2_pipeline.predict(ta)
            else:
                # Not all inputs are provided, use model1
                predict = model_pipeline.predict(ta)

            # Show prediction result
            if predict == 1:
                st.markdown("<h2 style='text-align:center;color:red'><mark>The cancer cells are malign. Patient has cancer.</mark></h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='text-align:center;color:green'><mark>Patient does not have cancer. The cancer cells are benign.</mark></h2>", unsafe_allow_html=True)
        
# Set background image
png_file = "/workspaces/Cancer-prediction/prediction/new detection/pic2.png"
page_config()
set_background(png_file)

# Run the main function
main()
