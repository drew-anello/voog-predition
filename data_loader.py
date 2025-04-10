"""
Data loading and preprocessing utilities for the VOOG Predictions App
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import config
import os
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AlpacaDataLoader:
    """
    Fetches historical data for VOOG from Alpaca API
    """

    def __init__(
        self, api_key=config.API_KEY, api_secret=config.API_SECRET
    ):
        """
        Initialize the Alpaca client with API keys
        """
        self.client = StockHistoricalDataClient(api_key, api_secret)
        logger.info("Initialized Alpaca Data Client")

    def fetch_historical_data(
        self,
        symbol=config.SYMBOL,
        start=config.START_DATE,
        end=config.END_DATE,
        timeframe=config.TIMEFRAME,
    ):
        """
        Fetch historical OHLCV data for the specified symbol

        Args:
            symbol: Stock symbol to fetch
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            timeframe: Time interval for the data

        Returns:
            DataFrame with historical data
        """
        logger.info(
            f"Fetching data for {symbol} from {start} to {end} with timeframe {timeframe}"
        )

        # Map timeframe string to TimeFrame object
        timeframe_map = {
            "1D": TimeFrame.Day,
            "1H": TimeFrame.Hour,
            "1Min": TimeFrame.Minute,
        }

        # Create request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_map.get(timeframe, TimeFrame.Day),
            start=pd.Timestamp(start, tz="America/New_York"),
            end=pd.Timestamp(end, tz="America/New_York"),
        )

        try:
            # Get the data
            bars = self.client.get_stock_bars(request_params)

            # Convert to DataFrame
            df = pd.DataFrame()
            if bars.data and symbol in bars.data:
                df = pd.DataFrame(bars.data[symbol])
                df.set_index("timestamp", inplace=True)
                df.index = pd.to_datetime(df.index)
                df = df[["open", "high", "low", "close", "volume"]]

                # Add additional features
                self._add_technical_indicators(df)

                logger.info(f"Successfully fetched {len(df)} rows of data")
            else:
                logger.error(f"No data returned for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        # Calculate rolling averages
        df["ma7"] = df["close"].rolling(window=7).mean()
        df["ma14"] = df["close"].rolling(window=14).mean()
        df["ma30"] = df["close"].rolling(window=30).mean()

        # Calculate daily returns
        df["returns"] = df["close"].pct_change()

        # Calculate volatility (standard deviation of returns)
        df["volatility"] = df["returns"].rolling(window=14).std()

        # Calculate trading volume moving average
        df["volume_ma7"] = df["volume"].rolling(window=7).mean()

        # Drop NaN values created by rolling windows
        df.dropna(inplace=True)

        return df


def prepare_dataset(
    data,
    sequence_length=config.SEQUENCE_LENGTH,
    prediction_horizon=config.HORIZON,
    train_split=config.TRAIN_SPLIT,
    val_split=config.VAL_SPLIT,
    test_split=config.TEST_SPLIT,
    batch_size=config.BATCH_SIZE,
):
    """
    Prepare the data for training by creating sequences and TensorFlow datasets

    Args:
        data: DataFrame with feature data
        sequence_length: Number of time steps to use as input
        prediction_horizon: Number of time steps to predict
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        batch_size: Batch size for training

    Returns:
        train_dataset, val_dataset, test_dataset, scaler
    """
    # Create a copy of the data
    df = data.copy()

    # Ensure the splits sum to 1
    assert abs(train_split + val_split + test_split - 1.0) < 1e-10, (
        "Splits must sum to 1"
    )

    # Separate features and target
    features = df[
        config.FEATURE_COLUMNS
        + ["ma7", "ma14", "ma30", "returns", "volatility", "volume_ma7"]
    ].values
    target = df[config.TARGET_COLUMN].values

    # Normalize the data with MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Target scaler (for the close price only)
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length - prediction_horizon + 1):
        X.append(features_scaled[i : i + sequence_length])
        y.append(
            target_scaled[
                i + sequence_length : i + sequence_length + prediction_horizon
            ]
        )

    X = np.array(X)
    y = np.array(y)

    # Calculate split indices
    n_samples = len(X)
    train_end = int(n_samples * train_split)
    val_end = train_end + int(n_samples * val_split)

    # Split the data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (
        train_dataset.shuffle(config.SHUFFLE_BUFFER)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info(
        f"Created datasets - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}"
    )

    # Save the scalers for later use
    with open(os.path.join(config.DATA_DIR, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(config.DATA_DIR, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)

    return train_dataset, val_dataset, test_dataset, scaler, target_scaler


def load_and_prepare_data():
    """
    Load data from Alpaca API and prepare it for training

    Returns:
        train_dataset, val_dataset, test_dataset, scaler, target_scaler, df
    """
    # Check if we have cached data
    data_path = os.path.join(config.DATA_DIR, f"{config.SYMBOL}_data.csv")

    if os.path.exists(data_path):
        logger.info(f"Loading cached data from {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Fetch new data
        data_loader = AlpacaDataLoader()
        df = data_loader.fetch_historical_data()

        if not df.empty:
            # Save the data
            df.to_csv(data_path)
        else:
            raise ValueError("Failed to fetch data from Alpaca API")

    # Prepare datasets
    train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = (
        prepare_dataset(df)
    )

    return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler, df
