import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def create_output_folder():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'emotion_recognition_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    subdirs = ['confusion_matrices', 'model_comparison', 'training_history', 
               'learning_curves', 'data_distribution']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir

def evaluate_model(y_true, y_pred, model_name, le=None, output_dir=None):
    if output_dir is None:
        output_dir = create_output_folder()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    target_names = le.classes_ if le is not None else [f"Class {i}" for i in range(len(np.unique(y_true)))]
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    results_file = os.path.join(output_dir, f'{model_name}_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"{model_name} Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (weighted): {f1_weighted:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=target_names))
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices', f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix
    }

def plot_model_comparison(model_metrics, le=None, output_dir=None):
    if output_dir is None:
        output_dir = create_output_folder()
        
    models = list(model_metrics.keys())
    accuracies = [metrics['accuracy'] for metrics in model_metrics.values()]
    f1_scores = [metrics['f1_weighted'] for metrics in model_metrics.values()]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison', 'model_comparison.png'))
    plt.close()

def plot_training_history(histories, model_names, output_dir=None):
    if output_dir is None:
        output_dir = create_output_folder()
        
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for hist, name in zip(histories, model_names):
        plt.plot(hist.history['accuracy'], label=f'{name} (train)')
        plt.plot(hist.history['val_accuracy'], label=f'{name} (val)')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for hist, name in zip(histories, model_names):
        plt.plot(hist.history['loss'], label=f'{name} (train)')
        plt.plot(hist.history['val_loss'], label=f'{name} (val)')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history', 'training_history.png'))
    plt.close()

def plot_learning_curves(histories, model_names, output_dir=None):
    if output_dir is None:
        output_dir = create_output_folder()
        
    metrics = ['accuracy', 'loss']
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        for hist, name in zip(histories, model_names):
            train_metric = hist.history[metric]
            val_metric = hist.history[f'val_{metric}']
            
            axes[i].plot(train_metric, label=f'{name} - Train')
            axes[i].plot(val_metric, label=f'{name} - Val')
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves', 'learning_curves.png'))
    plt.close()