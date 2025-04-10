import os
import dotenv
import logging
from datetime import datetime, timedelta
import tensorflow as tf

dotenv.load_dotenv()

# api configs
API_SECRET = os.getenv("API_SECRET")
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")

