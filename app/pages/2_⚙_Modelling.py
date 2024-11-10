import numpy as np
import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.model import get_model
from autoop.core.ml.metric import METRICS_CLASSIFICATION, METRICS_REGRESSION
from autoop.core.ml.metric import get_metric
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from typing import List

from autoop.functional.preprocessing import preprocess_features

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    A method to write helper text.
    :param text: str
    :return: None
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a "
                  "machine learning pipeline "
                  "to train a model on a dataset.")

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")


class PreprocessingHandler():
    """
    A class to handle the preprocessing of the data.
    """
    def __init__(self, session_state: dict) -> None:
        """
        A constructor for the PreprocessingHandler class.
        :param session_state: dict
        :return: None
        """
        self._model = None
        self.automl = automl
        self._desired_metrics = None
        self._dataset = None
        self._dataframe = None
        self._data_handler = session_state['data_handler']
        self._pipeline = None

    def select_ground_truth(self) -> str:
        """
        A method to select the ground truth column.
        :return: str
        """
        selection_ground_truth = st.selectbox(
            "Select the column with the data you want to predict:",
            options=self._dataframe.columns,
            placeholder="Select your ground truth...",
            index=None,
            key="select_ground_truth",
        )
        return selection_ground_truth

    def select_observations(self) -> list:
        """
        A method to select the observations columns.
        :return: list
        """
        selection_observations = st.multiselect(
            "Select your observations columns:",
            options=self._dataframe.columns,
            default=None,
            placeholder="Select one or more columns...",
            key="multiselect_observations"
        )
        return selection_observations

    def save_pipeline(self) -> None:
        """
        A method to save the pipeline.
        :return: None
        """
        self.automl.registry.register(self._pipeline)
        self._data_handler.save_in_registry(self._dataset)
        st.write("Pipeline saved.")

    def _can_load_existing_pipelines(self) -> bool:
        """
        A method to check if there are existing pipelines.
        :return: bool
        """
        return len(self.automl.registry.list(type="pipeline")) > 0

    def load_pipeline(self) -> None:
        """
        A method to load a pipeline.
        :return: None
        """
        pipelines = self.automl.registry.list(type="pipeline")
        pipe_name_to_id = {p.name: p.id for p in pipelines}
        pipeline_name = st.selectbox("Select your pipeline:",
                                     options=pipe_name_to_id.keys(),
                                     key="pipeline_select")
        pipe_art = automl.registry.get(pipe_name_to_id[pipeline_name])
        self._pipeline = Pipeline.from_artifact(pipe_art, automl)

    def _feature_selection(self, default_gt: str, default_obs: list) -> bool:
        """
        A method to select the features.
        :param default_gt: str
        :param default_obs: list
        :return: bool
        """
        self._selection_ground_truth = self._select_ground_truth(default_gt)
        self._selection_observations = self._select_observations(default_obs)
        self._handle_duplicate_features()
        return (len(self._selection_observations) != 0
                and self._selection_ground_truth is not None)

    def dataset_is_uploaded(self) -> bool:
        """
        A method to check if the dataset is uploaded.
        :return: bool
        """
        if (self._dataframe is None or
           'dataframe' not in st.session_state.keys()):
            st.write("Please upload your dataset in the \"Dataset\" page.")
            return False
        return True

    def _select_ground_truth(self, default: str = None) -> str:
        """
        A method to select the ground truth column.
        :param default: str
        :return: str
        """
        idx = 0
        if default is not None:
            idx = self._dataframe.columns.get_loc(default)
        selection_ground_truth = st.selectbox(
            "Select the column with the data you want to predict:",
            options=self._dataframe.columns,
            placeholder="Select your ground truth...",
            index=int(idx),
            key="select_ground_truth"
        )
        return selection_ground_truth

    def _select_observations(self, default: list = None) -> list:
        """
        A method to select the observations columns.
        :param default: list
        :return: list
        """
        selection_observations = st.multiselect(
            "Select your observations columns:",
            options=self._dataframe.columns,
            default=default,
            placeholder="Select one or more columns...",
            key="multiselect_observations"
        )
        return selection_observations

    def _handle_duplicate_features(self) -> list:
        """
        A method to handle duplicate features.
        :return: list
        """
        for observation in self._selection_observations:
            if observation == self._selection_ground_truth:
                st.markdown("You have selected the same column "
                            f"***{self._selection_ground_truth}*** "
                            "for both ground truth and observations.")
                st.markdown('''
            :red[**Please select another column for your observations!**]''')
                self._selection_observations.remove(observation)
        return self._selection_observations

    def _select_model(self) -> bool:
        """
        A method to select the model.
        :return: bool
        """
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

    def _select_metrics(self) -> bool:
        """
        A method to select the metrics.
        :return: bool
        """
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

    def validate_data(self) -> List[np.ndarray, np.ndarray]:
        """
        A method to validate the data.
        :return: Tuple[np.ndarray, np.ndarray]
        """
        data_x = np.asarray(self._dataframe[self._selection_observations])
        data_y = np.asarray([self._dataframe[self._selection_ground_truth]])
        if data_x.shape[0] != data_y.shape[0]:
            data_y.transpose()
        return data_x, data_y

    def run(self) -> None:
        """
        A method to run the data handler.
        :return: None
        """
        if self.dataset_is_uploaded:
            self._dataframe = self._data_handler.df
            self._dataset = self._data_handler.dataset
            if not isinstance(self._dataset, Dataset):
                raise TypeError(f"{type(self._dataset)} is not Dataset ")
            st.write(self._dataframe.head())

            if self._can_load_existing_pipelines():
                if st.button("Load Pipeline"):
                    self.load_pipeline()

            defaultY = None
            defaultX = None
            if self._pipeline is not None:
                defaultY = self._pipeline.target_feature.name
                defaultX = [f.name for f in self._pipeline.input_features]
            if self._feature_selection(defaultY, defaultX):
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
                            target_feature=detect_feature_types(
                                self._y_data)[0],
                            split=data_split)

                        if st.button("Save Pipeline"):
                            self.save_pipeline()

                        if st.button("Predict"):
                            st.divider()

                            prep_x = preprocess_features(
                                self._pipeline.input_features, self._dataset)
                            data_x = None
                            for item in prep_x:
                                if data_x is None:
                                    data_x = item[1]
                                else:
                                    data_x = np.concatenate(
                                        [data_x, item[1]], axis=1)

                            data_y = np.asarray(self._dataframe[
                                self._selection_ground_truth])

                            train_x, test_x, train_y, tes_y = train_test_split(
                                data_x, data_y, train_size=data_split / 100,
                                shuffle=False)

                            self._model.fit(train_x, train_y)
                            y_pred = self._model.predict(test_x)

                            metric_results = []
                            for metric in self._desired_metrics:
                                metric_results.append(
                                    {metric.name: metric.evaluate(
                                        tes_y, y_pred)})

                            st.header("Predictions:")
                            st.write(y_pred[:5])

                            observations_columns_count = len(
                                self._selection_observations)
                            if (self._model.type == "regression"
                                    and observations_columns_count == 1):
                                plt.figure(figsize=(10, 6))
                                plt.scatter(test_x, tes_y, color='blue',
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
                                unique_classes = np.unique(tes_y)

                                for class_value in unique_classes:
                                    class_indices = (
                                        tes_y == class_value).ravel()
                                    plt.scatter(
                                        test_x[class_indices, 0],
                                        test_x[class_indices, 1],
                                        label=f"Class {class_value}",
                                        s=40, edgecolor='k'
                                    )

                                plt.xlabel(
                                    f"{self._selection_observations[0]}")
                                plt.ylabel(
                                    f"{self._selection_observations[1]}")
                                plt.title(
                                    "2D Scatter Plot of Classification Data")
                                plt.legend()
                                st.pyplot(plt.gcf())
                            else:
                                fig, axes = plt.subplots(
                                    nrows=1, ncols=observations_columns_count,
                                    figsize=(15, 5))
                                if observations_columns_count == 1:
                                    # Single subplot case
                                    axes.hist(test_x[:, 0], bins=15, alpha=0.7)
                                    axes.set_title(
                                        self._selection_observations[0])
                                else:
                                    # Multiple subplots case
                                    for i in range(observations_columns_count):
                                        axes[i].hist(test_x[:, i], bins=15,
                                                     alpha=0.7)
                                        axes[i].set_title(
                                            self._selection_observations[i])

                                plt.suptitle("Distribution of Each Feature")
                                st.pyplot(fig)

                            st.header("**Metrics**")
                            unique_classes = np.unique(tes_y)
                            for metric_result in metric_results:
                                for metric, result in metric_result.items():
                                    if metric == "Confusion Matrix":
                                        st.markdown("***Confusion Matrix:***")
                                        metrics.ConfusionMatrixDisplay(
                                            confusion_matrix=result,
                                            display_labels=unique_classes
                                        ).plot()
                                        st.pyplot(plt.gcf())
                                    else:
                                        st.markdown(f"**{metric}:** {result}")


if "data_handler" not in st.session_state:
    st.write("Please upload your dataset in the \"Dataset\" page.")
else:
    if st.session_state['data_handler'].df is not None:
        modelling = PreprocessingHandler(st.session_state)
        modelling.run()
    else:
        st.write("Please upload your dataset in the \"Dataset\" page.")
