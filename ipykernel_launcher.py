import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Set page title and icon
st.set_page_config(page_title="RSF Prediction App for the Overall Survival of Pediatric Wilms' Tumor", page_icon=":bar_chart:")

# Add a title and subheader
st.title("RSF Prediction App for the Overall Survival of Pediatric Wilms' Tumor")
st.subheader("Enter patient information and click Submit to get the prediction")

# Create a sidebar for input fields
with st.sidebar:
    st.header("Input Information")
    Stage = st.selectbox("Stage", [1, 2, 3, 4])
    Age = st.number_input("Age")
    Distant_Metastasis = st.checkbox("Distant Metastasis")
    Preoperative_Chemotherapy = st.checkbox("Preoperative Chemotherapy")
    Perirenal_Fat_Invasion = st.checkbox("Perirenal Fat Invasion")
    Preoperative_tumor_rupture = st.checkbox("Preoperative tumor rupture")
    Radiotherapy = st.checkbox("Radiotherapy")
    submit_button = st.button("Submit")

# If button is pressed
if submit_button:
    try:
        # Load the saved model from the file
        with open('./rsf.pkl', 'rb') as f:
            clf = pickle.load(f)

        # Store inputs into dataframe
        X = pd.DataFrame([[Perirenal_Fat_Invasion, Preoperative_tumor_rupture, 
                           Preoperative_Chemotherapy, Stage, Age,
                           Distant_Metastasis, Radiotherapy]], 
                         columns=['Perirenal_Fat_Invasion', 'Preoperative_tumor_rupture', 
                                  'Preoperative_Chemotherapy', 'Stage', 'Age',
                                  'Distant_Metastasis', 'Radiotherapy'])

        surv = clf.predict_survival_function(X, return_array=True)

        for i, s in enumerate(surv):
            plt.step(range(125), s, where="post", label=str(i))
        plt.ylabel("Survival probability")
        plt.xlabel("Survival time (mo)")
        plt.legend(loc='upper right', labels=['' for _ in range(len(surv))] )
        plt.grid(True)

        # Output survival probabilities
        st.pyplot(plt)
        st.success("Survival probabilities plotted successfully.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

   
