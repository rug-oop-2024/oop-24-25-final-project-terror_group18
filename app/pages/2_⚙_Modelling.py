import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
# print(datasets)

data = pd.read_csv(r"C:\Users\Iva\Downloads\wine+quality\winequality-red.csv")
st.write(data.columns)
# features_selection = st.column_config.SelectboxColumn(label=None, width=None, help=None, disabled=None, required=None, default=None, options=None)

# data = pd.read_csv(datasets)


for column in data.columns:
    st.selectbox(
        f"Select an option for {column}", 
        (column)
    )

# # Display the selected column's data
# st.write(f"You selected {selected_column}")
# st.write(datasets[selected_column])
# # if features_selection is not None:
# #     dataset = automl.registry.get(features_selection)
# #     st.write(dataset.read())


