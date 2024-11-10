from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Exception raised when a path is not found
    """
    def __init__(self, path: str) -> None:
        """
        The constructor for the NotFoundError class.
        :param path: str path that was not found
        :return: None
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract class for storage
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """The LocalStorage class stores data in a local directory."""
    def __init__(self, base_path: str = "./assets") -> None:
        """
        The constructor for the LocalStorage class.
        :param base_path: str base path to store data
        :return: None
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        Returns:
            None
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        Returns:
            None
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path)
                for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Assert that a path exists
        Args:
            path (str): Path to check
        Returns:
            None
        """
        path = self._join_path(path)
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join a path to the base path
        Args:
            path (str): Path to join
        Returns:
            str: Joined path
        """
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
