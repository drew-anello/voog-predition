"""
TensorFlow model definitions for VOO price prediction
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Input,
    Bidirectional,
    GRU,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
import os
import config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_lstm_model(sequence_length, n_features, output_size):
    """
    Create a stacked LSTM model for time series forecasting

    Args:
        sequence_length: Number of time steps in input sequences
        n_features: Number of features in input data
        output_size: Number of time steps to predict

    Returns:
        Compiled TensorFlow model
    """
    model = Sequential(
        [
            # First LSTM layer with return sequences for stacking
            LSTM(
                units=config.MODEL_PARAMS["lstm_units"][0],
                return_sequences=True,
                input_shape=(sequence_length, n_features),
            ),
            # Dropout for regularization
            Dropout(config.MODEL_PARAMS["dropout_rate"]),
            # Second LSTM layer
            LSTM(units=config.MODEL_PARAMS["lstm_units"][1], return_sequences=False),
            # Dropout for regularization
            Dropout(config.MODEL_PARAMS["dropout_rate"]),
            # Dense hidden layer
            Dense(units=config.MODEL_PARAMS["dense_units"][0], activation="relu"),
            # Output layer
            Dense(units=output_size, activation="linear"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config.MODEL_PARAMS["learning_rate"]),
        loss=config.MODEL_PARAMS["loss"],
        metrics=config.MODEL_PARAMS["metrics"],
    )

    logger.info(f"Created LSTM model with {model.count_params()} parameters")
    model.summary()

    return model


def create_bidirectional_lstm_model(sequence_length, n_features, output_size):
    """
    Create a bidirectional LSTM model for time series forecasting

    Args:
        sequence_length: Number of time steps in input sequences
        n_features: Number of features in input data
        output_size: Number of time steps to predict

    Returns:
        Compiled TensorFlow model
    """
    model = Sequential(
        [
            # Bidirectional LSTM layer
            Bidirectional(
                LSTM(units=config.MODEL_PARAMS["lstm_units"][0], return_sequences=True),
                input_shape=(sequence_length, n_features),
            ),
            # Dropout for regularization
            Dropout(config.MODEL_PARAMS["dropout_rate"]),
            # Second Bidirectional LSTM layer
            Bidirectional(
                LSTM(units=config.MODEL_PARAMS["lstm_units"][1], return_sequences=False)
            ),
            # Dropout for regularization
            Dropout(config.MODEL_PARAMS["dropout_rate"]),
            # Dense hidden layer
            Dense(units=config.MODEL_PARAMS["dense_units"][0], activation="relu"),
            # Output layer
            Dense(units=output_size, activation="linear"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config.MODEL_PARAMS["learning_rate"]),
        loss=config.MODEL_PARAMS["loss"],
        metrics=config.MODEL_PARAMS["metrics"],
    )

    logger.info(
        f"Created Bidirectional LSTM model with {model.count_params()} parameters"
    )
    model.summary()

    return model


def create_cnn_lstm_model(sequence_length, n_features, output_size):
    """
    Create a hybrid CNN-LSTM model for time series forecasting

    Args:
        sequence_length: Number of time steps in input sequences
        n_features: Number of features in input data
        output_size: Number of time steps to predict

    Returns:
        Compiled TensorFlow model
    """
    # Input layer
    input_layer = Input(shape=(sequence_length, n_features))

    # CNN layers for feature extraction
    conv1 = Conv1D(filters=64, kernel_size=3, activation="relu")(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=3, activation="relu")(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    # LSTM layers for temporal dependencies
    lstm1 = LSTM(units=config.MODEL_PARAMS["lstm_units"][0], return_sequences=True)(
        pool2
    )
    drop1 = Dropout(config.MODEL_PARAMS["dropout_rate"])(lstm1)
    lstm2 = LSTM(units=config.MODEL_PARAMS["lstm_units"][1])(drop1)
    drop2 = Dropout(config.MODEL_PARAMS["dropout_rate"])(lstm2)

    # Dense layers for prediction
    dense1 = Dense(units=config.MODEL_PARAMS["dense_units"][0], activation="relu")(
        drop2
    )
    output_layer = Dense(units=output_size, activation="linear")(dense1)

    # Create and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=config.MODEL_PARAMS["learning_rate"]),
        loss=config.MODEL_PARAMS["loss"],
        metrics=config.MODEL_PARAMS["metrics"],
    )

    logger.info(f"Created CNN-LSTM model with {model.count_params()} parameters")
    model.summary()

    return model


def get_callbacks(model_name):
    """
    Create callbacks for model training

    Args:
        model_name: Name of the model for saving

    Returns:
        List of callbacks
    """
    # Create the checkpoint path
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}.h5")

    # Define callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Model checkpoint to save best model
        ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        # Reduce learning rate when model plateaus
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        # TensorBoard for visualization
        TensorBoard(
            log_dir=os.path.join(config.TENSORBOARD_DIR, model_name),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
    ]

    return callbacks


def train_model(model, train_dataset, val_dataset, model_name="voog_lstm"):
    """
    Train the model with the given datasets

    Args:
        model: TensorFlow model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_name: Name of the model for saving

    Returns:
        Training history
    """
    logger.info(f"Training model {model_name}")

    # Get callbacks
    callbacks = get_callbacks(model_name)

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the final model
    model.save(os.path.join(config.MODELS_DIR, f"{model_name}_final.h5"))

    return history


def create_model(
    model_type="lstm",
    sequence_length=config.SEQUENCE_LENGTH,
    n_features=None,
    output_size=config.HORIZON,
):
    """
    Factory function to create the specified model type

    Args:
        model_type: Type of model to create ('lstm', 'bidirectional', 'cnn_lstm')
        sequence_length: Number of time steps in input sequences
        n_features: Number of features in input data
        output_size: Number of time steps to predict

    Returns:
        Compiled TensorFlow model
    """
    # If n_features is not specified, use the default calculated value
    if n_features is None:
        # Base features + technical indicators
        n_features = len(config.FEATURE_COLUMNS) + 6  # 6 technical indicators

    # Create the specified model type
    if model_type == "lstm":
        return create_lstm_model(sequence_length, n_features, output_size)
    elif model_type == "bidirectional":
        return create_bidirectional_lstm_model(sequence_length, n_features, output_size)
    elif model_type == "cnn_lstm":
        return create_cnn_lstm_model(sequence_length, n_features, output_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
