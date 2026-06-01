import os
from dotenv import load_dotenv

#Load the .env file
load_dotenv()

#Hyderabad, Sindh's coordinates
LATITUDE  = 25.3960
LONGITUDE = 68.3578
TIMEZONE  = "Asia/Karachi"
 
#Output Paths
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
 
#Raw JSON: original untouched API responses
MERGED_JSON_PATH = os.path.join(RAW_DATA_DIR, "aqi_merged.json")


MONGO_URI        = os.getenv("MONGO_URI")        
MONGO_DB_NAME    = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")
MONGO_MODELS_COLLECTION = os.getenv("MONGO_MODELS_COLLECTION")