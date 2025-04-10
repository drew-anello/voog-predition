"""
Script to train the VOOG prediction model using TensorFlow
"""

import os
import json
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import config
from data_loader import load_and_prepare_data
from model import create_model, train_model
from visualization import plot_training_history

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Main function to train the model

    Args:
        args: Command-line arguments
    """
    logger.info("Starting model training process...")
    logger.info(f"Using model type: {args.model_type}")

    # Check TensorFlow version
    logger.info(f"TensorFlow version: {tf.__version__}")

    # Check for available GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Available GPUs: {len(gpus)}")
        for gpu in gpus:
            logger.info(f"  {gpu}")
    else:
        logger.info("No GPUs available. Using CPU.")

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load and prepare data
    logger.info("Loading and preparing data...")
    train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler, df = (
        load_and_prepare_data()
    )

    # Get input shape from dataset
    for x, y in train_dataset.take(1):
        sequence_length = x.shape[1]
        n_features = x.shape[2]
        output_size = y.shape[1]
        logger.info(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Create the model
    model = create_model(
        model_type=args.model_type,
        sequence_length=sequence_length,
        n_features=n_features,
        output_size=output_size,
    )

    # Generate a unique model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.SYMBOL}_{args.model_type}_{timestamp}"

    # Train the model
    logger.info(f"Training model {model_name}...")
    history = train_model(model, train_dataset, val_dataset, model_name=model_name)

    # Plot training history
    logger.info("Plotting training history...")
    history_plot = plot_training_history(history, model_name)
