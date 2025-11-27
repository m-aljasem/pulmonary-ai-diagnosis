"""
Training script for chest X-ray multi-label classification.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

from model import build_chest_xray_model


def create_generators(df_train, df_val, df_test, image_size=(224, 224), batch_size=32):
    """Create data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1/255.0)
    
    # Adjust based on your data structure
    return None, None, None


def train_model(epochs=50, batch_size=32):
    """Main training function."""
    print("Training chest X-ray classification model...")
    
    # Load data
    # df = pd.read_csv('../data/train.csv')
    # df_train, df_temp = train_test_split(df, test_size=0.2)
    # df_val, df_test = train_test_split(df_temp, test_size=0.5)
    
    # Create generators
    # train_gen, val_gen, test_gen = create_generators(df_train, df_val, df_test)
    
    # Build model
    model = build_chest_xray_model(input_shape=(224, 224, 3), num_classes=14)
    
    # Ensure models directory exists
    models_dir = Path('../models')
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / 'chest_xray_model.h5'
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(str(weights_path), save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    # Train
    # history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    # print(f"✓ Best model weights saved to {weights_path}")
    
    print("Training complete! (Enable training code and generators when data is ready.)")


if __name__ == '__main__':
    train_model()

