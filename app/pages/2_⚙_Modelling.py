import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline "
                  "to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

data = pd.read_csv(r"C:\Users\Iva\Downloads\Life Expectancy Data.csv")
st.write(data.head())
# features_selection = st.column_config.SelectboxColumn(label=None, width=None, help=None, disabled=None, required=None, default=None, options=None)

# data = pd.read_csv(datasets)


selection_ground_truth = st.selectbox(
        "Select the column with the data you want to predict:",
        options=data.columns,
        placeholder="Select your ground truth...",
        index=None,
    )

selection_observations = st.multiselect(
    "Select your observations columns:",
    options=data.columns,
    default=None,          # No default selection
    placeholder="Select one or more columns..."
)

train_test_split = st.slider("Select your train/test split", 0, 100)

st.divider()

st.markdown("*Before you continue, these are your selections so far:*")
# Y DATA
if selection_ground_truth is None:
    st.markdown('''
    :red[**Please select something as your ground truth!**]''')
else:
    st.markdown(f"You have selected the ***{selection_ground_truth}*** "
                "column as your ground truth.")

# X DATA
        # raise error when diff types of cat/cont mix data columns are selected??
if len(selection_observations) == 0:
    st.markdown('''
    :red[**Please select at least one column as your observations!**]''')
else:
    st.write(f"You have selected the ***{selection_observations}*** "
             "column as your observations.")

# TRAIN/TEST SPLIT
        # if train_test_split == 0 or 100: can we use this? 
        # or should we rise errors
st.write(f"You have decided to use ***{train_test_split}%*** of your "
         f"data for training and ***{100 - train_test_split}%*** for testing.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - train_test_split)/100)