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
from sklearn import metrics

from autoop.functional.preprocessing import preprocess_features

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
                    # st.write(self._model)
                    # st.write(type(self._model))

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

                            # data_x = np.asarray(
                            #     self._dataframe[self._selection_observations])
                            # data_y = np.asarray(
                            #     [self._dataframe[self._selection_ground_truth]]).transpose()
                            

                            prep_x = preprocess_features(self._pipeline.input_features, self._dataset)
                            data_x = None
                            for item in prep_x:
                                #st.write(item[1])
                                if data_x is None:
                                    data_x = item[1]
                                else:
                                    data_x = np.concatenate([data_x, item[1]], axis=1)
                                st.write(f"dataX is {data_x[:5]}")
                                # for i in range(len(item[1])):
                                #     st.write(item[1][i])
                                #     data_x[i].append(item[1][i])
                            #data_x = np.concatenate([x[1] for x in prep_x], axis=1)
                            # [x[1] if x[2]['type'] != 'OneHotEncoder' else np.array([x[1]]) for x in prep_x]
                            # data_x = np.asarray(data_x)
                            #st.write(data_x)

                            prep_y = preprocess_features([self._pipeline.target_feature],
                                                         self._dataset)
                            #st.write(item for item in prep_y)
                            data_y = prep_y[0][1]
                            st.write(f"dataY is {data_y[:5]}")

                            train_x, test_x, train_y, test_y = train_test_split(
                                data_x, data_y, train_size=data_split / 100,
                                shuffle=False)

                            # st.write("Shape of train_x:", train_x.shape)
                            # st.write("Shape of train_y:", train_y.shape)

                            # st.write(type(self._model))
                            self._model.fit(train_x, train_y)
                            y_pred = self._model.predict(test_x)

                            metric_results = []
                            for metric in self._desired_metrics:
                                metric_results.append(
                                    {metric.name: metric.evaluate(
                                        test_y, y_pred)})

                            st.write("Predictions:")
                            # maybe make the x columnS appear as well??
                            st.write(y_pred)

                            # st.write(self._model.type)
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
                            # else:
                            #     fig, axes = plt.subplots(
                            #         nrows=1, ncols=observations_columns_count,
                            #         figsize=(15, 5))
                            #     for i in range(observations_columns_count):
                            #         axes[i].hist(
                            #             test_x[:, i], bins=15, alpha=0.7)
                            #         axes[i].set_title(f"Feature {i+1}")
                            #     plt.suptitle("Distribution of Each Feature")
                            #     st.pyplot(fig)
                            # plt.scatter(test_x, y_pred)
                            # st.pyplot(plt.gcf())




                            # slope, intercept, r, p, std_err = stats.linregress(test_x, test_y)

                            # def myfunc(x):
                            #     return slope * x + intercept

                            # mymodel = list(map(myfunc, x))

                            # plt.scatter(test_x, test_y)
                            # plt.plot(test_x, mymodel)
                            # # st.write(pd.DataFrame(y_pred).head())


                            st.markdown("**Metrics**")
                            # for metric, result in metric_results:
                            #     st.markdown(f"{metric}: {result}")
                            for metric_result in metric_results:
                                for metric, result in metric_result.items():
                                    st.markdown(f"**{metric}:** {result}")
                                    if metric == "Confusion Matrix":
                                        st.markdown("***Confusion Matrix:***")
                                        metrics.ConfusionMatrixDisplay(
                                            confusion_matrix=result,
                                            display_labels=[0, 1]).plot()
                                        st.pyplot(plt.gcf())
                                    
                                    

if "data_handler" not in st.session_state:
    st.write("Please upload your dataset in the \"Dataset\" page.")
else:
    # leave that for now because without it we have error!
    if st.session_state['data_handler'].df is not None:
        modelling = PreprocessingHandler(st.session_state)
        modelling.run()
    else:
        st.write("Please upload your dataset in the \"Dataset\" page.")
