import unittest

from autoop.core.database import Database
from autoop.core.storage import LocalStorage
import random
import tempfile


class TestDatabase(unittest.TestCase):
    """
    Test the Database class
    """
    def setUp(self) -> None:
        """
        Set up the test environment
        :return: None
        """
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self) -> None:
        """
        Test the initialization of the Database class
        :return: None
        """
        self.assertIsInstance(self.db, Database)

    def test_set(self) -> None:
        """
        Test the set method of the Database class
        :return: None
        """
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self) -> None:
        """
        Test the delete method of the Database class
        :return: None
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistance(self) -> None:
        """
        Test the persistence of the Database class
        :return: None
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self) -> None:
        """
        Test the refresh method of the Database class
        :return: None
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self) -> None:
        """
        Test the list method of the Database class
        :return: None
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # collection should now contain the key
        self.assertIn((key, value), self.db.list("collection"))
