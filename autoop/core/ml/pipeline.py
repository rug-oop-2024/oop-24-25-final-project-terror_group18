from copy import deepcopy
from typing import List
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import get_model
from autoop.core.ml.model.base_model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline(Artifact):
    """
    The Pipeline class.
    """
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 name: str = "Untitled",
                 *args, **kwargs
                 ) -> None:
        """
        The Pipeline class constructor.
        :param metrics: List[Metric]
        :param dataset: Dataset
        :param model: Model
        :param input_features: List[Feature]
        :param target_feature: Feature
        :param split: float
        :return: None
        """
        super().__init__(
            name=name, asset_path=name, type='pipeline', *args, **kwargs)
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        self._update_data()
        if self._model is not None:
            if ((target_feature.type == "categorical") and (
               model._type != "classification")):
                raise ValueError("Model type must be classification "
                                 "for categorical target feature")
            if ((target_feature.type == "continuous") and (
               model._type != "regression")):
                raise ValueError("Model type must be regression "
                                 "for continuous target feature")

    @property
    def input_features(self) -> List[Feature]:
        """
        The getter method for the input features.
        :return: List[Feature]
        """
        return deepcopy(self._input_features)

    @property
    def target_feature(self) -> Feature:
        """
        The getter method for the target feature.
        :return: Feature
        """
        return deepcopy(self._target_feature)

    def __str__(self) -> str:
        """The string representation of the pipeline.
        :return: str
        """
        return f"""
Pipeline(
    model={self._model._type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        The getter method for the model object.
        :return: Model
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the pipeline execution
        to be saved.
        :return: List[Artifact]
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        if self._model is not None:
            artifacts.append(self._model.to_artifact(
                name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _update_data(self) -> None:
        """Update the data property of the pipeline artifact.
        :return: None
        """
        liberal_arts_ids = [art.id for art in self.artifacts]

        pipeline_data = {
            "artifact_IDs": liberal_arts_ids,
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        self.save(pickle.dumps(pipeline_data))

    @staticmethod
    def from_artifact(artifact: Artifact, automl: AutoMLSystem) -> 'Pipeline':
        """
        The method that converts an artifact to a pipeline.
        :param artifact: Artifact
        :param automl: AutoMLSystem
        :return: Pipeline
        """
        print(artifact.type)
        pipeline_data = pickle.loads(artifact.data.decode())
        ds = None
        md = None
        for id in pipeline_data["artifact_IDs"]:
            print(id)
            art = automl.registry.get(id)
            if art is None:
                continue
            if art.type == 'dataset':
                df = art.read()
                ds = Dataset.from_dataframe(df, art.name, art.asset_path)
            elif art.type == 'model':
                md = get_model(art.name)

        return Pipeline(
            metrics=None,
            dataset=ds,
            model=md,
            input_features=pipeline_data['input_features'],
            target_feature=pipeline_data['target_feature'],
            split=pipeline_data["split"]
        )

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        The method that registers an artifact in the pipeline.
        :param name: str
        :param artifact: Artifact
        :return: None
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        The method that preprocesses the features.
        :return: None
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by
        # feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (
            feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """
        The method that splits the data into training and testing sets.
        :return: None
        """
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [vector[:int(split * len(vector))]
                         for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):]
                        for vector in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        The method that concatenates the input vectors into a single vector.
        :param vectors: List[np.array]
        :return: np.array
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        The method that fits the training data to the model.
        :return: None
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        The method that evaluates the model on the test data.
        :return: None
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> None:
        """
        The method that executes the pipeline.
        :return: None
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        X = self._compact_vectors(self._test_X)
        self._predictions = self._model.predict(X)
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
