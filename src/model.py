"""Multi-label classification model for chest X-rays"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


def build_chest_xray_model(input_shape=(224, 224, 3), num_classes=14):
    """Build DenseNet121 model for multi-label chest X-ray classification."""
    base = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='sigmoid')(x)  # Multi-label
    
    model = Model(inputs=base.input, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

