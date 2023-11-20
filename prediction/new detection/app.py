import pandas as pd
import numpy as np
import streamlit as st
import pickle
import base64
dict_att={"radius_mean":None,"texture_mean":None,"perimeter_mean":None,"area_mean":None,"smoothness_mean":None,"compactness_mean":None,"concavity_mean":None,"concave points_mean":None,"symmetry_mean":None,"fractal_dimension_mean":None,"radius_se":None,"texture_se":None,"perimeter_se":None,"area_se":None,"smoothness_se":None,"compactness_se":None,"concavity_se":None,"concave_points_se":None,"symmetry_se":None,"fractal_dimension_se":None,"radius_worst":None,"texture_worst":None,"perimeter_worst":None,"area_worst":None,"smoothness_worst":None,"compactness_worst":None,"concavity_worst":None,"concave points_worst":None,"symmetry_worst":None,"fractal_dimension_worst":None}
k=list(dict_att.keys())
l1=[]
f=open("model.pkl","rb")
data=pickle.load(f)
f2=open("model2.pkl","rb")
data2=pickle.load(f2)
def page_config():
    st.set_page_config(
    page_title="Breast cancer predictor",
    page_icon="🧊",
    layout="wide",
        
)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    width:cover;
    height:cover;
    border: 2px solid;

    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
                                                                                      
                                                                                      
def image():
    st.image("pic2.png")
def main():
    submit=0
    flag=0
    st.markdown("<h1 style='text-align:center; color:black;'>Breast Cancer Predictor</h1>",unsafe_allow_html=True)
    st.write("   ")
    st.write("<h4 style='color:black;'>Please connect this app with your cytology lab to help diagnose breast cancer from your sample.This app predicts using a machine learning model whether a breast mass is benign or malignant based on measurements it receives from your cytosis lab.Enter the details of parameters you have.</h4>",unsafe_allow_html=True)
    st.write("<h3 style='color:grey'>Select parameters you want to enter from below:</h3>",unsafe_allow_html=True)
    l1=st.multiselect("",k)
    st.write("\n\n\n\n")
    st.sidebar.title("Enter the details of cell here:-")
    for i in l1:
        x=st.sidebar.number_input("Enter the value of {}:".format(i))
        dict_att[i]=x
    button1=st.sidebar.button("Submit")
    if button1==True:
        submit=1
        
    
    
    with st.container() as co: 
        test=list(dict_att.values())
        test=test[0:29]
        ta=np.array(test)
        ta=ta.reshape(1,-1)
        for i in test:
            if i==None:
                flag+=1
        if flag==0:
            predict=data2.predict(ta)
        else:
            predict=data.predict(ta)
        if submit==1:
            if predict==1:
                st.write("<h2 style='text-align:center;color:black'><mark style='color:red'>The cancer cells are malign.Patient have cancer.</mark></h2>",unsafe_allow_html=True)
            else:
                st.write("<h2 style='text-align:center;color:black'><mark style='color:orange'>Patient does not have cancer.The cancer cells are benign.</mark></h2>",unsafe_allow_html=True)

png_file="pic2.png"
page_config()
set_background(png_file)
main()    
                   
