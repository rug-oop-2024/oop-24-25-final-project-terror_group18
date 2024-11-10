from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """
    A class for registering and retrieving artifacts.
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        The constructor for the ArtifactRegistry class.

        :param database: The database to use for storing metadata.
        :param storage: The storage to use for storing the artifact data.
        :return: None
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register an artifact in the registry.

        :param artifact: The artifact to register.
        :return: None
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        List all the artifacts in the registry.

        :param type: str type of the artifact
        :return: A list of all the artifacts in the registry.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Get an artifact from the registry.

        :param artifact_id: str id of the artifact
        :return: Artifact
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an artifact from the registry.

        :param artifact_id: str id of the artifact
        :return: None
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Singleton class for the AutoML system
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        The constructor for the AutoMLSystem class.

        :param storage: The storage to use for storing the artifact data.
        :param database: The database to use for storing metadata.
        :return: None
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """
        Get the instance of the AutoMLSystem class.
        :return: The instance of the AutoMLSystem class.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Get the registry of the AutoMLSystem class.
        :return: ArtifactRegistry
        """
        return self._registry
