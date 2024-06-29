import streamlit as st

st.set_page_config(
    page_title="msiFlow",
    page_icon="",
)

st.logo("imgs/msiFlow_logo.svg", icon_image="imgs/msiFlow_logo.svg")

st.write("# Welcome to msiFlow!")

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    msiFlow is a collection of automated workflows for reproducible and scalable multimodal mass spectrometry imaging 
    (MSI) and immunofluorescence (IF) microscopy data processing and analysis. \n
    Select a workflow from the sidebar to adjust the parameters and run the workflow. \n
    Informations on how to structure the input files and set the parameters are provided on 
    [GitHub](https://github.com/Immunodynamics-Engel-Lab/msiflow).
    """
)
