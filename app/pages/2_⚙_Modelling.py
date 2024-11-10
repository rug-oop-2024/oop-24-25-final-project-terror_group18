import numpy as np
import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from app.core.ui_utils import DataHandler
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.model import get_model
from autoop.core.ml.metric import METRICS_CLASSIFICATION, METRICS_REGRESSION
from autoop.core.ml.metric import get_metric
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline "
                  "to train a model on a dataset.")

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")


class PreprocessingHandler():

    def __init__(self, session_state):
        self._model = None
        self.automl = automl
        self._desired_metrics = None
        self._dataset = None
        self._dataframe = None
        self._data_handler = session_state['data_handler']
        self._pipeline = None

    def select_ground_truth(self):
        selection_ground_truth = st.selectbox(
            "Select the column with the data you want to predict:",
            options=self._dataframe.columns,
            placeholder="Select your ground truth...",
            index=None,
            key="select_ground_truth",
        )
        return selection_ground_truth

    def select_observations(self):
        selection_observations = st.multiselect(
            "Select your observations columns:",
            options=self._dataframe.columns,
            default=None,
            placeholder="Select one or more columns...",
            key="multiselect_observations"
        )

        return selection_observations

    def save_pipeline(self):
        for artifact in self._pipeline.artifacts:
            self.automl.registry.register(artifact)
            self._data_handler.save_in_registry(self._dataset)
        st.write("Pipeline saved.")

    def _can_load_existing_pipelines(self):
        if self.automl.registry.list(type="pipeline") is None:
            return False
        return True

    def load_pipeline(self):
        self._pipeline = st.selectbox("Select your pipeline:",
                                      options=self.automl.registry.list(
                                          type="pipeline"),
                                      key="pipeline_select")

    def _feature_selection(self):
        self._selection_ground_truth = self._select_ground_truth()
        self._selection_observations = self._select_observations()
        self._handle_duplicate_features()
        # st.write(':red[WARNING: All rows with NaN values will be dropped.]')
        # self._dataframe.dropna(subset=self._selection_observations, inplace=True)
        # self._dataframe.dropna(subset=[self._selection_ground_truth], inplace=True)
        return (len(self._selection_observations) != 0 and
                self._selection_ground_truth is not None)

    def dataset_is_uploaded(self):
        if self._dataframe is None or 'dataframe' not in st.session_state.keys():
            st.write("Please upload your dataset in the \"Dataset\" page.")
            return False
        return True

    def _select_ground_truth(self):
        selection_ground_truth = st.selectbox(
            "Select the column with the data you want to predict:",
            options=self._dataframe.columns,
            placeholder="Select your ground truth...",
            index=None,
            key="select_ground_truth",
        )
        return selection_ground_truth

    def _select_observations(self):
        selection_observations = st.multiselect(
            "Select your observations columns:",
            options=self._dataframe.columns,
            default=None,
            placeholder="Select one or more columns...",
            key="multiselect_observations"
        )
        return selection_observations

    def _handle_duplicate_features(self):
        # ERROR: when we select the same column for both ground
        # truth and observations
        for observation in self._selection_observations:
            if observation == self._selection_ground_truth:
                st.markdown("You have selected the same column "
                            f"***{self._selection_ground_truth}*** "
                            "for both ground truth and observations.")
                st.markdown('''
            :red[**Please select another column for your observations!**]''')
                self._selection_observations.remove(observation)
        return self._selection_observations

    def _select_model(self):
        types_options = {
            "categorical": ["classification", CLASSIFICATION_MODELS],
            "numerical": ["regression", REGRESSION_MODELS]}
        for i, feature_type in enumerate(detect_feature_types(self._y_data)):
            self._feature_type = feature_type.type
            self._model_choice = st.selectbox(
                f"Select your {types_options[self._feature_type][0]} model:",
                options=types_options[self._feature_type][1],
                placeholder="Select your model...",
                index=None,
                key=f"classification_model_selectbox_{i}"
            )
            if self._model_choice is None:
                st.markdown(
                    ''':red[*You have not selected a model yet!*]''')
                return False
            return True

    def _select_metrics(self):
        metrics_types = {'categorical': METRICS_CLASSIFICATION,
                         'numerical': METRICS_REGRESSION}
        for i, feature_type in enumerate(detect_feature_types(self._y_data)):
            self._feature_type = feature_type.type
            self._metric_choice = st.multiselect(
                "Select your metrics:",
                options=metrics_types[self._feature_type],
                default=None,
                placeholder="Select one or more metrics...",
                key=f"multiselect_metrics_{i}"
            )
        if len(self._metric_choice) == 0:
            st.markdown(
                ''':red[*You have not selected a metric yet!*]''')
            return False
        return True

    def validate_data(self):
        data_x = np.asarray(self._dataframe[self._selection_observations])
        data_y = np.asarray([self._dataframe[self._selection_ground_truth]])
        if data_x.shape[0] != data_y.shape[0]:
            data_y.transpose()
        return data_x, data_y

    def run(self):
        if self.dataset_is_uploaded:
            self._dataframe = self._data_handler.df
            self._dataset = self._data_handler.dataset
            if not isinstance(self._dataset, Dataset):
                raise TypeError(f"{type(self._dataset)} is not Dataset ")
            st.write(self._dataframe.head())

            if self._can_load_existing_pipelines():
                if st.button("Load Pipeline"):
                    self.load_pipeline()

            if self._feature_selection():
                self._X_data = Dataset.from_dataframe(
                    data=self._dataframe[self._selection_observations],
                    name="Observations Data",
                    asset_path="Observations.csv")
                self._y_data = Dataset.from_dataframe(
                    data=self._dataframe[self._selection_ground_truth],
                    name="Ground Truth Data",
                    asset_path="Ground Truth.csv")
                data_split = st.slider("Select your train/test split",
                                       0, 100, value=80)

                if self._select_model():
                    self._model = get_model(self._model_choice)
                    st.write(self._model)
                    st.write(type(self._model))

                    if self._select_metrics():
                        self._desired_metrics = []
                        for metric in self._metric_choice:
                            self._desired_metrics.append(get_metric(metric))

                        st.markdown("*Before you continue, "
                                    "these are your selections so far:*")
                        st.markdown(
                            "***Ground Truth:*** "
                            f"*{self._selection_ground_truth}*")
                        st.markdown(
                            "***Observations:*** "
                            f"*{self._selection_observations}*")
                        st.markdown(f"***Split:*** {data_split}% Train,"
                                    f" {100 - data_split}% Test")
                        st.markdown(f"***Model:*** *{self._model_choice}*")
                        st.markdown(f"***Metrics:*** *{self._metric_choice}*")

                        self._pipeline = Pipeline(
                            metrics=self._desired_metrics,
                            dataset=self._dataset,
                            model=self._model,
                            input_features=detect_feature_types(self._X_data),
                            target_feature=detect_feature_types(self._y_data)[0],
                            split=data_split)

                        if st.button("Save Pipeline"):
                            self.save_pipeline()

                        # pages = {
                        #     "Instructions": "./pages/0_✅_Instructions.py",
                        #     "Dataset": "./pages/1_📊_Datasets.py",
                        #     "Modelling": "./pages/2_⚙_Modelling.py",
                        #     "Predictions": "./pages/3_Predictions.py"
                        # }

                        if st.button("Predict"):
                            # selected_page = "Predictions"

                            st.divider()
                            # prediction_results = self._pipeline.execute()
                            # st.session_state["pipeline_results"] = prediction_results
                            # st.write(prediction_results)

                            # if st.button("Predict"):
                            #     st.session_state["prediction_results"] = prediction_results
                            #     page_file = pages[selected_page]
                            #     st.switch_page(page_file)

                            data_x = np.asarray(
                                self._dataframe[self._selection_observations])
                            data_y = np.asarray(
                                [self._dataframe[self._selection_ground_truth]]).transpose()
                            train_x, test_x, train_y, test_y = train_test_split(
                                data_x, data_y, train_size=data_split/100,
                                shuffle=False)

                            st.write("Shape of train_x:", train_x.shape)
                            st.write("Shape of train_y:", train_y.shape)

                            st.write(type(self._model))
                            self._model.fit(train_x, train_y)
                            y_pred = self._model.predict(test_x)
                            # metric_results = []
                            # for metric in self._desired_metrics:
                            #     metric_results.append({metric.name : metric.evaluate(test_y, y_pred)})

                            st.write("Predictions:")
                            st.write(y_pred)

                            st.write(self._model.type)
                            observations_columns_count = len(
                                self._selection_observations)
                            if (self._model.type == "regression"
                               and observations_columns_count == 1):
                                plt.figure(figsize=(10, 6))
                                plt.scatter(test_x, test_y, color='blue',
                                            label='Actual Test Data')
                                plt.plot(test_x, y_pred, color='red',
                                         linewidth=2,
                                         label='Prediction Line of Best Fit')
                                plt.xlabel("Feature")
                                plt.ylabel("Target")
                                plt.title("Regression: Test Data with "
                                          "Prediction Line of Best Fit")
                                plt.legend()
                                st.pyplot(plt.gcf())
                            elif (self._model.type == "classification"
                                  and observations_columns_count == 2):
                                plt.figure(figsize=(10, 6))
                                unique_classes = np.unique(test_y)  # Get unique classes from test_y

                                # Loop over each unique class to plot with a different color
                                for class_value in unique_classes:
                                    class_indices = (test_y == class_value).ravel()  # Indices of samples belonging to the current class
                                    plt.scatter(
                                        test_x[class_indices, 0], test_x[class_indices, 1],
                                        label=f"Class {class_value}", s=40, edgecolor='k'
                                    )

                                plt.xlabel("Feature 1")
                                plt.ylabel("Feature 2")
                                plt.title("2D Scatter Plot of Classification Data")
                                plt.legend()
                                st.pyplot(plt.gcf())
                            else:
                                fig, axes = plt.subplots(
                                    nrows=1, ncols=observations_columns_count,
                                    figsize=(15, 5))
                                for i in range(observations_columns_count):
                                    axes[i].hist(
                                        test_x[:, i], bins=15, alpha=0.7)
                                    axes[i].set_title(f"Feature {i+1}")
                                plt.suptitle("Distribution of Each Feature")
                                st.pyplot(fig)
                            # plt.scatter(test_x, y_pred)
                            # st.pyplot(plt.gcf())

                            # slope, intercept, r, p, std_err = stats.linregress(test_x, test_y)

                            # def myfunc(x):
                            #     return slope * x + intercept

                            # mymodel = list(map(myfunc, x))

                            # plt.scatter(test_x, test_y)
                            # plt.plot(test_x, mymodel)
                            # # st.write(pd.DataFrame(y_pred).head())


                            # st.write("Metrics")
                            # for metric, result in metric_results:
                            #     st.markdown(f"{metric}: {result}")



# def dataset_is_uploaded():
#     if 'dataframe' not in st.session_state.keys():
#         return False
#     return True


# def select_ground_truth(dataframe):
#     selection_ground_truth = st.selectbox(
#         "Select the column with the data you want to predict:",
#         options=dataframe.columns,
#         placeholder="Select your ground truth...",
#         index=None,
#         key="select_ground_truth",
#     )
#     return selection_ground_truth


# def select_observations(dataframe):
#     selection_observations = st.multiselect(
#         "Select your observations columns:",
#         options=dataframe.columns,
#         default=None,
#         placeholder="Select one or more columns...",
#         key="multiselect_observations"
#     )

#     return selection_observations



# def handle_duplicate_features(selection_ground_truth, selection_observations):
#     # ERROR: when we select the same column for both ground
#     # truth and observations
#     for observation in selection_observations:
#         if observation == selection_ground_truth:
#             st.markdown("You have selected the same column "
#                         f"***{selection_ground_truth}*** "
#                         "for both ground truth and observations.")
#             st.markdown('''
#         :red[**Please select another column for your observations!**]''')
#             selection_observations.remove(observation)
#     return selection_observations


# def check_is_none(obj):
#     if obj is None:
#         return True
#     return False

# def check_is_empty(lst):
#     if len(lst) == 0:
#         return True
#     return False


# def split_data():
#     return st.slider("Select your train/test split", 0, 100)


# automl = AutoMLSystem.get_instance()
# datasets = automl.registry.list(type="dataset")

# if not dataset_is_uploaded():
#     st.write("Please upload your dataset in the \"Dataset\" page.")
# else:
#     dataframe = st.session_state['dataframe']
#     dataset = st.session_state['data_handler'].dataset
#     st.write(dataframe.head())

#     selection_ground_truth = select_ground_truth(dataframe)
#     selection_observations = select_observations(dataframe)

#     selection_observations = handle_duplicate_features(
#         selection_ground_truth, selection_observations)
#     # if check_is_empty(selection_observations):
#     #     select_observations(dataframe)

#     data_split = split_data()

#     predict_button = False
#     st.divider()
#     st.markdown("*Before you continue, these are your selections so far:*")
#     # Y DATA
#     if check_is_none(selection_ground_truth):
#         st.markdown('''
#         :red[**Please select something as your ground truth!**]''')
#     else:
#         st.markdown(f"You have selected the ***{selection_ground_truth}*** "
#                     "column as your ground truth.")
#         Y_data = Dataset.from_dataframe(data=dataframe[selection_ground_truth],
#                                         name="Ground Truth Data",
#                                         asset_path="Ground Truth.csv")
#         st.write(dataframe[selection_ground_truth].head())

#     # X DATA
#     if check_is_empty(selection_observations):
#         st.markdown('''
#         :red[**Please select at least one column as your observations!**]''')
#     else:
#         st.write(f"You have selected the ***{selection_observations}*** "
#                  "column as your observations.")
#         X_data = Dataset.from_dataframe(data=dataframe[selection_observations],
#                                         name="Observations Data",
#                                         asset_path="Observations.csv")
#         st.write(dataframe[selection_observations].head())

#     # TRAIN/TEST SPLIT
#     st.write(f"You have decided to use ***{data_split}%*** of your "
#              f"data for training and ***{100 - data_split}%*** for testing.")
#     # !!!!!!!!!!!!!!!! turn into fractionn
#     data_split /= 100

#     st.divider()
#     if not check_is_none(selection_ground_truth) and not check_is_empty(selection_observations):
#         model_choice = None
#         metric_choice = None
#         # feature = detect_feature_types(Y_data)
#         # if feature.type == "categorical":
#         #         model_choice = st.selectbox(
#         #             "Select your classification model:",
#         #             options=CLASSIFICATION_MODELS,
#         #             placeholder="Select your model...",
#         #             index=None,
#         #             key=f"classification_model_selectbox_{i}"
#         #         )

#         #         metric_choice = st.multiselect(
#         #             "Select your metrics:",
#         #             options=METRICS_CLASSIFICATION,
#         #             default=None,          # No default selection
#         #             placeholder="Select one or more metrics...",
#         #             key=f"multiselect_metrics_{i}"
#         #         )

#         # elif feature.type == "numerical":
#         #     model_choice = st.selectbox(
#         #         "Select your regression model:",
#         #         options=REGRESSION_MODELS,
#         #         placeholder="Select your model...",
#         #         index=None,
#         #         key=f"regression_model_selectbox_{i}"
#         #     )

#         #     metric_choice = st.multiselect(
#         #         "Select your metrics:",
#         #         options=METRICS_REGRESSION,
#         #         default=None,          # No default selection
#         #         placeholder="Select one or more metrics...",
#         #         key=f"multiselect_metrics_{i}"
#         #     )
#         # else:
#         #     st.markdown(
#         #         ''':red[*You have not selected a model or metrics yet!*]''')
#         for i, feature in enumerate(detect_feature_types(Y_data)):
#             if feature.type == "categorical":
#                 model_choice = st.selectbox(
#                     "Select your classification model:",
#                     options=CLASSIFICATION_MODELS,
#                     placeholder="Select your model...",
#                     index=None,
#                     key=f"classification_model_selectbox_{i}"
#                 )

#                 metric_choice = st.multiselect(
#                     "Select your metrics:",
#                     options=METRICS_CLASSIFICATION,
#                     default=None,  # No default selection
#                     placeholder="Select one or more metrics...",
#                     key=f"multiselect_metrics_{i}"
#                 )

#             elif feature.type == "numerical":
#                 model_choice = st.selectbox(
#                     "Select your regression model:",
#                     options=REGRESSION_MODELS,
#                     placeholder="Select your model...",
#                     index=None,
#                     key=f"regression_model_selectbox_{i}"
#                 )

#                 metric_choice = st.multiselect(
#                     "Select your metrics:",
#                     options=METRICS_REGRESSION,
#                     default=None,          # No default selection
#                     placeholder="Select one or more metrics...",
#                     key=f"multiselect_metrics_{i}"
#                 )
#             else:
#                 st.markdown(
#                     ''':red[*You have not selected a model or metrics yet!*]''')

#             model = get_model(model_choice)

#             desired_metrics = []
#             for metric in metric_choice:
#                 desired_metrics.append(get_metric(metric))

#             if model_choice is not None:
#                 if len(desired_metrics) != 0:
#                     predict_button = True
#                     pipeline = Pipeline(model=model,
#                                         dataset=dataset,
#                                         # dataset=st.session_state['dataset_id'],
#                                         input_features=X_data,
#                                         target_feature=Y_data,
#                                         split=data_split,
#                                         metrics=metric_choice)
#                     if st.button("Save Pipeline"):
#                         try:
#                             for artifact in pipeline.artifacts:
#                                 automl.registry.register(artifact)
#                             st.write("Pipeline saved.")
#                         except Exception as e:
#                             st.write(e) #maybe we dont need the try except anymore...
#                         """for artifact in pipeline.artifacts:
#                             automl.registry.register(artifact)
#                             DataHandler.save_in_registry(dataset)
#                         st.write("Pipeline saved.")"""

#             #     pipeline = automl.pipeline(model, X_data, Y_data, data_split)


#     # save model & model_id before pipeline; load model by id


#     pages = {
#             "Instructions": "./pages/0_✅_Instructions.py",
#             "Dataset": "./pages/1_📊_Datasets.py",
#             "Modelling": "./pages/2_⚙_Modelling.py",
#             "Predictions": "./pages/3_Predictions.py"
#         }

#     selected_page = "Predictions"


#     if predict_button:
#         st.divider()
#         predict_results = pipeline.execute()
#         st.markdown("*Before you continue, these are your selections so far:*")
#         st.markdown(f"***Model:*** {model_choice}")
#         st.markdown(f"***Metrics:*** {metric_choice}")
#         if st.button("Predict"):
#             st.session_state["pipeline_results"] = predict_results
#             page_file = pages[selected_page]
#             st.switch_page(page_file)
#         #if st.button("Predict", on_click=printtt):
#             # st.switch_page("Predictions")
#             #model.fit(X_data, Y_data)... train/test

#     # pipeline = automl.pipeline(model, X_data, Y_data, data_split)

#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     X, y, test_size=(100 - data_split)/100)


#     # selected_page = "Dataset"
#     # # Button to switch page
#     # switch_page = st.button("Switch page")
#     # if switch_page:
#     #     # Switch to the selected page
#     #     page_file = pages[selected_page]
#     #     st.switch_page(page_file):
#     #     # ????st.write(f"You are now on page: {selected_page}")


if "data_handler" not in st.session_state:
    st.write("Please upload your dataset in the \"Dataset\" page.")
else:
    # leave that for now because without it we have error!
    if st.session_state['data_handler'].df is not None:
        modelling = PreprocessingHandler(st.session_state)
        modelling.run()
    else:
        st.write("Please upload your dataset in the \"Dataset\" page.")
