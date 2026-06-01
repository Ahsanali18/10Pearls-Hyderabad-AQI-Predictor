"""
src/db/database_connection.py
=======================
Single MongoDB connection used by all pipelines and dashboard.
Never create MongoClient anywhere else — always import get_db().
"""

from pymongo import MongoClient
from config.settings import MONGO_URI, MONGO_DB_NAME
import math


def get_db_client():
    """
    Returns (client, db) tuple.
    Always call client.close() when done in scripts.
    """
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    return client, client[MONGO_DB_NAME]


def nan_to_none(value):
    """Convert NaN and Inf to None for MongoDB compatibility."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
