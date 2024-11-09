import streamlit as st
import pandas as pd
from autoop.core.ml.dataset import Dataset
# from autoop.core.ml.model.regression import RidgeRegression
from sklearn.model_selection import train_test_split



st.set_page_config(page_title="Predictions", page_icon="🗠")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# 🗠 Predictions")
write_helper_text("These are your predictions for the given dataset:")

if "pipeline_results" not in st.session_state.keys():
    st.write("Please select a pipeline first.")
else:
    st.write(st.session_state["pipeline_results"])


# st.set_page_config(page_title="Modelling", page_icon="📈")
# dataframe = pd.read_csv(r"C:\Users\Iva\Downloads\Life Expectancy Data.csv")
# dataset = Dataset.from_dataframe(data=dataframe, name="Life Expectancy Data",
#                                  asset_path="Life Expectancy Data.csv")
# st.write(dataframe.head())

# selection_ground_truth = st.selectbox(
#         "Select the column with the data you want to predict:",
#         options=dataframe.columns,
#         placeholder="Select your ground truth...",
#         index=None,
#         key="select_ground_truth",
#     )
# y = dataframe[selection_ground_truth].head()


# selection_observations = st.multiselect(
#     "Select your observations columns:",
#     options=dataframe.columns,
#     default=None,          # No default selection
#     placeholder="Select one or more columns...",
#     key="multiselect_observations"
# )
# X = dataframe[selection_observations].head()

# # model = RidgeRegression()

# # model.fit(dataframe[selection_observations], dataframe[selection_ground_truth])
# data_split = st.slider("Select your train/test split", 0, 100)
# st.write(data_split)
# st.write(X)
# st.write(y)


# # pages = {
# #     "Instructions": "./pages/0_✅_Instructions.py",
# #     "Dataset": "./pages/1_📊_Datasets.py",
# #     "Modelling": "./pages/2_⚙_Modelling.py",
# #     "Predictions": "./pages/3_Predictions.py"
# # }
# # selected_page = "Dataset"
# # # Button to switch page
# # switch_page = st.button("Switch page")
# # if switch_page:
# #     # Switch to the selected page
# #     page_file = pages[selected_page]
# #     st.switch_page(page_file):
# #     # ????st.write(f"You are now on page: {selected_page}")

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_iris
# import shap
# import matplotlib.pyplot as plt
# # THIS EXAMPLE HAS ERRORS BUT IT IS AN IDEA
# # Load a dataset and split into train/test sets
# data = load_iris()
# X, y = data.data, data.target
# feature_names = data.feature_names
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Create a SHAP explainer for the trained model
# explainer = shap.TreeExplainer(model)

# # Calculate SHAP values for the test set
# shap_values = explainer.shap_values(X_test)

# # Summary plot for global feature importance
# shap.summary_plot(shap_values, X_test, feature_names=feature_names)
# plt.show()  # Optionally, add plt.show() to ensure the plot displays in certain environments


# # Force plot for the first prediction in the test set (individual prediction explanation)
# shap.initjs()
# shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=feature_names)