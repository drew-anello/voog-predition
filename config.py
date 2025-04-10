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

# data configs
SYMBOL = "VOOG"
START_DATE = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")
TIMEFRAME = "1D"

# data processing 
SEQUENCE_LENGTH = 60 #num of days used for predicition input
PREDICTION_DAYS = 5 #num of days to prdicit ahead
FEATURE_COLS = ["open", "high", "low", "close", "volume"]
TARGET_COL = "close"
TRAIN_SPLIT = 0.8 #training data
VAL_SPLIT = 0.15 #validation data
TEST_SPLIT = 0.05 #test data

# model params 
MODEL_PARAMS = {
    "lstm_units" : [64, 64],
    "dense_units" : 64, 
    "dropout_rate" : 0.2,
    "learning_rate" : 0.001,
    "loss" : "mse",
    "metrics" : ["mse", "mae", "mape"]
}

# Training Params
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001


# paths 
DATA_DIR = os.path.join(os.getcwd(), "data")
MODEL_DIR = os.path.join(os.getcwd(), "models")
LOG_DIR = os.path.join(os.getcwd(), "logs")
TENSORBOARD_DIR = os.path.join(os.getcwd(), "tensorboard")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Check if GPU is available
USE_GPU = tf.config.list_physical_devices("GPU")
if USE_GPU:
    print("GPU is available. Using GPU for training.")
    # Configure TensorFlow to use GPU memory growth
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("GPU is not available. Using CPU for training.")

