from copy import deepcopy

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")


class DataHandler:

    def __init__(self):
        self.df = None
        self._dataset_name_to_id = {dt.name: dt.id for dt in datasets}
        self._options = ["UPLOAD"] + list(self._dataset_name_to_id.keys())
        self._option = None
        self._file_path = None
        self._file = None

    def _choose_file(self):
        self._option = st.selectbox("Choose Dataset", self._options)
        if self._option is None:
            st.write('Please select something')
        if self._option == "UPLOAD":
            self._file = st.file_uploader("Choose a file", type='csv')
            if self._file is not None:
                self._file_path = self._file.name
                self.df = pd.read_csv(self._file)
        else:
            self._dataset_id = self._dataset_name_to_id[self._option]
            self._file = automl.registry.get(self._dataset_id)
            if self._file is not None:
                self._file_path = self._file.asset_path
                self.df = self._file.read()

    def _handle(self):
        self.display()
        self._dataset = Dataset.from_dataframe(
            data=self.df, name=self._file_path.removesuffix('.csv'),
            asset_path=self._file_path)
        st.session_state['dataframe'] = self.df
        #st.session_state['df_dataset'] = self._dataset
        #st.session_state['dataset_id'] = self._dataset_id

    def dataset(self):
        return deepcopy(self._dataset)

    def display(self):
        st.write(self.df.head())
        st.write(f"You chose {self._file_path.removesuffix('.csv')}")

    @staticmethod
    def save_in_registry(dataset: Dataset) -> bool:
        if dataset.asset_path not in [x.asset_path for x in datasets]:
            automl.registry.register(dataset)
            return True
        return False

    def run(self):
        self._choose_file()
        if self._file is not None:
            self._handle()
            if self._option == "UPLOAD":
                if st.button("Save Dataset"):
                    if self.save_in_registry(self._dataset):
                        st.write("Dataset saved.")
                    else:
                        st.write(f"Dataset with path {self._dataset.asset_path} already exists. "
                                 f"Please select another dataset.")
