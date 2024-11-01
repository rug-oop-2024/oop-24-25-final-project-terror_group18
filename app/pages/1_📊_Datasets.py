from glob import glob
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# option = st.selectbox(glob("**/*.csv", recursive=True))
option = st.selectbox("Dataset", glob("**/*.csv", recursive=True))
if option is None:
    st.write('Please select something')

df = pd.read_csv(option)
st.write(df.head())
options = st.multiselect(df.columns)
