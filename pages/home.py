import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

def app():
    st.title('Home')

    st.write('This is the `home page` of this multi-page app.')

    st.write('In this app, we will be building a simple classification model using the Iris dataset.')

    st.write("""
        # Simple iris Flower Prediction App This app predicts the iris Flower type!
        """)

   

