from glob import glob
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

# option = st.selectbox(glob("**/*.csv", recursive=True))
option = st.selectbox("Choose Dataset", ["UPLOAD", datasets])
if option == "UPLOAD":
    option = st.file_uploader("Choose a file", type='csv')
if option is not None:
    df = pd.read_csv(option)
    st.write(df.head())
    # options = st.multiselect(df.columns)
    if st.button("Save Dataset"):
        automl.registry.register(
            Dataset.from_dataframe(data=df,
                                   name=option.name.removesuffix('.csv'),
                                   asset_path=option.name))
        st.write("Dataset saved.")
else:
    st.write('Please select something')
