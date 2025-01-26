import os
import numpy as np
import librosa
import librosa.display
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from data_utils import load_ravdess_tess, prepare_data
from models import build_hybrid_model, HuggingFaceModel
from evaluation import analyze_dataset_distribution, evaluate_model, plot_enhanced_metrics, plot_model_comparison, plot_training_history, plot_learning_curves, create_output_folder

class AdditionalMetrics(Callback):
    def __init__(self, validation_data):
        super(AdditionalMetrics, self).__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        for i in range(y_pred.shape[1]):
            logs[f'class_{i}_accuracy'] = accuracy_score(
                np.argmax(self.validation_data[1], axis=1) == i,
                np.argmax(y_pred, axis=1) == i
            )

def main():
    output_dir = create_output_folder()
    df = load_ravdess_tess('data/Audio_Speech_Actors_01-24', 'data/TESS/TESS')
    analyze_dataset_distribution(df, output_dir)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    X_train, X_test, y_train, y_test, le = prepare_data(df)
    
    model_metrics = {}
    
    # HuggingFace model
    benchmark = HuggingFaceModel()
    benchmark_X_train = benchmark.prepare_data(train_df['path'])
    benchmark_X_test = benchmark.prepare_data(test_df['path'])
    benchmark_pred = benchmark.predict(benchmark_X_test)
    model_metrics['HuggingFace'] = evaluate_model(y_test, benchmark_pred, "HuggingFace Model", le, output_dir)
    
    # Hybrid model
    input_shape = (X_train.shape[1], X_train.shape[2])
    hybrid_model = build_hybrid_model(input_shape, len(le.classes_))
    
    val_data = (X_train[-len(X_train)//5:], y_train[-len(y_train)//5:])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
        AdditionalMetrics(validation_data=val_data)
    ]
    
    history = hybrid_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    hybrid_pred = hybrid_model.predict(X_test)
    model_metrics['Hybrid'] = evaluate_model(y_test, hybrid_pred, "Hybrid Model", le, output_dir)
    
    plot_model_comparison(model_metrics, le, output_dir)
    plot_training_history([history], ['Hybrid'], output_dir)
    plot_learning_curves([history], ['Hybrid'], output_dir)
    plot_enhanced_metrics(history, y_test, hybrid_pred, le, output_dir)

if __name__ == "__main__":
    main()