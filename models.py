import torch
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout, 
    Flatten, GRU, Bidirectional, Concatenate, BatchNormalization, Multiply
)
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class HuggingFaceModel:
    def __init__(self, model_name="Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        
    def prepare_data(self, file_paths):
        features = []
        for path in file_paths:
            audio, _ = librosa.load(path, sr=16000)
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            features.append(inputs)
        return features
    
    def predict(self, X):
        predictions = []
        for feature in X:
            with torch.no_grad():
                logits = self.model(**feature).logits
                predicted_label = torch.argmax(logits, dim=1).item()
                predictions.append(predicted_label)
        return np.array(predictions)

def build_hybrid_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    
    # CNN Branch
    conv1 = Conv1D(64, 5, activation='relu', padding='same')(x)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(4)(bn1)
    drop1 = Dropout(0.3)(pool1)
    
    conv2 = Conv1D(128, 3, activation='relu', padding='same')(drop1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(2)(bn2)
    drop2 = Dropout(0.4)(pool2)
    
    conv3 = Conv1D(256, 3, activation='relu', padding='same')(drop2)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(2)(bn3)
    flat_cnn = Flatten()(pool3)
    
    # CNN intermediate predictions
    cnn_hidden = Dense(128, activation='relu')(flat_cnn)
    cnn_logits = Dense(num_classes)(cnn_hidden)
    
    # GRU Branch
    gru1 = Bidirectional(GRU(64, return_sequences=True))(x)
    bn_gru1 = BatchNormalization()(gru1)
    drop_gru1 = Dropout(0.3)(bn_gru1)
    
    gru2 = Bidirectional(GRU(128))(drop_gru1)
    bn_gru2 = BatchNormalization()(gru2)
    
    # GRU intermediate predictions
    gru_hidden = Dense(128, activation='relu')(bn_gru2)
    gru_logits = Dense(num_classes)(gru_hidden)
    
    # Stacking layer
    stacked_features = Concatenate()([
        flat_cnn,
        bn_gru2,
        cnn_logits,
        gru_logits
    ])
    
    # Meta-classifier
    meta1 = Dense(256, activation='relu')(stacked_features)
    meta_drop1 = Dropout(0.5)(meta1)
    meta2 = Dense(128, activation='relu')(meta_drop1)
    meta_drop2 = Dropout(0.4)(meta2)
    
    outputs = Dense(num_classes, activation='softmax')(meta_drop2)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model