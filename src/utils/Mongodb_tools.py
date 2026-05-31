import sys
import os
from pymongo import MongoClient
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import *

class MONGODB_TOOLS():
    def __init__(self):
        self.db_name = MONGO_DB_NAME
        self.db_uri = MONGODB_URI

        # Build the client once
        self._client = MongoClient(self.db_uri)

    def get_db_connection(self, admin=False):
        """Return a pymongo Database object.

        The `admin` flag is kept for API compatibility but has no effect
        in MongoDB (authentication is handled at connection-URI level).
        """
        return self._client[self.db_name]

    # Alias kept for call-sites that used get_db_connection2
    def get_db_connection2(self, admin=False):
        return self.get_db_connection(admin=admin)

    def insert_dataframe_to_mongo(self, dataframe, collection_name):
        """Bulk-insert a pandas DataFrame into a MongoDB collection."""
        db = self.get_db_connection()
        dataframe = dataframe.fillna('')
        records = dataframe.to_dict(orient='records')
        if records:
            db[collection_name].insert_many(records)

    # Keep backward-compatible name
    insert_dataframe_to_postgre = insert_dataframe_to_mongo
