import streamlit as st
import pandas as pd
import pickle
import os
# Set page title and icon
st.set_page_config(page_title="RSF Prediction App", page_icon=":bar_chart:")

# Add a title and subheader
st.title("RSF Prediction App")
st.subheader("Enter patient information and click Submit to get the prediction")

# Create a sidebar for input fields
with st.sidebar:
    st.header("Input Information")
    Stage = st.number_input("Stage")
    Age = st.number_input("Age")
    Distant_Metastasis = st.checkbox("Distant Metastasis")
    Preoperative_Chemotherapy = st.checkbox("Preoperative Chemotherapy")
    Perirenal_Fat_Invasion = st.checkbox("Perirenal Fat Invasion")
    Preoperative_tumor_rupture = st.checkbox("Preoperative tumor rupture")
    Radiotherapy = st.checkbox("Radiotherapy")
    submit_button = st.button("Submit")

try:
    with open('/mount/src/mechinelearning/your_pickle_file.pkl', 'rb') as f:
        clf = pickle.load(f)
except Exception as e:
    print("An error occurred while loading the pickle file:", e)
# If button is pressed
if submit_button:
    # Load the saved model from the file
    
    # Load the saved model from the file
    with open('/mount/src/mechinelearning/rsf.pkl', 'rb') as f:
        clf = pickle.load(f)

    
    # Store inputs into dataframe
    X = pd.DataFrame([[Perirenal_Fat_Invasion, Preoperative_tumor_rupture, 
                       Preoperative_Chemotherapy, Stage, Age,
                       Distant_Metastasis, Radiotherapy]], 
                     columns=['Perirenal_Fat_Invasion', 'Preoperative_tumor_rupture', 
                              'Preoperative_Chemotherapy', 'Stage', 'Age',
                              'Distant_Metastasis', 'Radiotherapy'])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.success(f"This instance is predicted as: {prediction}")

   
