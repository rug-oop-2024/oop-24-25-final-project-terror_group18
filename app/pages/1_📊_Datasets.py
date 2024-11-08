from glob import glob
import streamlit as st
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

# option = st.selectbox(glob("**/*.csv", recursive=True))
dataset_name_to_id = {dt.name: dt.id for dt in datasets}
options = ["UPLOAD"] + list(dataset_name_to_id.keys())
option = st.selectbox("Choose Dataset", options)
file_path = None

if option == "UPLOAD":
    file = st.file_uploader("Choose a file", type='csv')
else:
    file = automl.registry.get(dataset_name_to_id[option])


if file is not None:
    if isinstance(file, UploadedFile):
        file_path = file.name
        df = pd.read_csv(file)
    else:
        file_path = file.asset_path
        df = file.read()

    st.session_state['dataframe'] = df
    st.write(df.head())
    st.write(f"You chose {file_path.removesuffix('.csv')}")
    df_dataset = Dataset.from_dataframe(
        data=df, name=file_path.removesuffix('.csv'), asset_path=file_path)
    st.session_state['df_dataset']= df_dataset

    if option == "UPLOAD":
        if st.button("Save Dataset"):
            if df_dataset.asset_path not in [x.asset_path for x in datasets]:
                # st.write(df_dataset.__str__())
                automl.registry.register(df_dataset)
                st.write("Dataset saved.")
            else:
                st.write(f"Dataset with path {df_dataset.asset_path} already exists. "
                         f"Please select another dataset.")
else:
    st.write('Please select something')
