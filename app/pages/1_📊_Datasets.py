from glob import glob
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

# option = st.selectbox(glob("**/*.csv", recursive=True))
options = ["UPLOAD"] + [x.name for x in datasets]
option = st.selectbox("Choose Dataset", options)
if option == "UPLOAD":
    option = st.file_uploader("Choose a file", type='csv')


if option is not None:
    df = pd.read_csv(option)
    st.write(df.head())
    st.write(option.name.removesuffix('.csv'))
    if st.button("Save Dataset"):
        df_dataset = Dataset.from_dataframe(
            data=df, name=option.name.removesuffix('.csv'), asset_path=option.name)
        st.write(df_dataset.__str__())
        automl.registry.register(df_dataset)
        st.write("Dataset saved.")
else:
    st.write('Please select something')
