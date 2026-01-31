"""
Module d'Entraînement pour NIDS MindSpore
Huawei ICT Competition 2025-2026

Gère l'entraînement, la validation, et l'évaluation du modèle
avec support pour Ascend 910 via CANN.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, Model
from mindspore.train.callback import (
    ModelCheckpoint, CheckpointConfig, 
    LossMonitor, TimeMonitor
)
from mindspore.nn.metrics import Accuracy, Precision, Recall, F1
import numpy as np
import yaml
import os
from datetime import datetime
import json


class NIDSTrainer:
    """
    Entraîneur pour le modèle NIDS ResNet-LSTM.
    
    Features:
    - Entraînement sur Ascend/GPU/CPU
    - Early stopping
    - Checkpointing automatique
    - Logging des métriques
    - Visualisation de la matrice de confusion
    """
    
    def __init__(self, model, config_path: str = "config/config.yaml"):
        """
        Args:
            model: Modèle ResNetLSTMNIDS
            config_path: Chemin vers la configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = model
        self.setup_context()
        self.setup_training()
        
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = []
    
    def setup_context(self):
        """Configure le contexte MindSpore pour Ascend/GPU."""
        device_target = self.config['mindspore']['device_target']
        device_id = self.config['mindspore']['device_id']
        mode = self.config['mindspore']['mode']
        
        ms_mode = context.GRAPH_MODE if mode == "GRAPH_MODE" else context.PYNATIVE_MODE
        
        context.set_context(
            mode=ms_mode,
            device_target=device_target,
            device_id=device_id
        )
        
        print(f"🔧 Context: {device_target} (Device {device_id}), Mode: {mode}")
    
    def setup_training(self):
        """Configure les composants d'entraînement."""
        # Loss function
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        
        # Optimizer
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler_config = self.config['training']['scheduler']
        if scheduler_config['type'] == 'CosineAnnealing':
            self.lr_scheduler = nn.CosineDecayLR(
                min_lr=scheduler_config['eta_min'],
                max_lr=lr,
                decay_steps=scheduler_config['T_max']
            )
        
        # Metrics
        self.metrics = {
            'accuracy': Accuracy(),
            'precision': Precision(),
            'recall': Recall(),
            'f1': F1()
        }
    
    def train_epoch(self, train_dataset):
        """
        Entraîne le modèle pour une epoch.
        
        Args:
            train_dataset: Dataset d'entraînement MindSpore
            
        Returns:
            Loss moyenne de l'epoch
        """
        self.model.set_train(True)
        
        total_loss = 0.0
        num_batches = 0
        
        # Forward function
        def forward_fn(data, label):
            logits = self.model(data)
            loss = self.loss_fn(logits, label)
            return loss, logits
        
        # Gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
        
        for batch in train_dataset.create_dict_iterator():
            features = batch['features']
            labels = batch['label']
            
            # Forward + backward
            (loss, logits), grads = grad_fn(features, labels)
            self.optimizer(grads)
            
            total_loss += loss.asnumpy()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, val_dataset):
        """
        Évalue le modèle sur le dataset de validation.
        
        Args:
            val_dataset: Dataset de validation
            
        Returns:
            Dictionnaire des métriques
        """
        self.model.set_train(False)
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_dataset.create_dict_iterator():
            features = batch['features']
            labels = batch['label']
            
            # Forward
            logits = self.model(features)
            loss = self.loss_fn(logits, labels)
            
            predictions = ops.Argmax(axis=1)(logits)
            
            all_predictions.extend(predictions.asnumpy().tolist())
            all_labels.extend(labels.asnumpy().tolist())
            
            total_loss += loss.asnumpy()
            num_batches += 1
        
        # Calculer les métriques
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_predictions == all_labels)
        
        # Métriques par classe
        num_classes = self.config['dataset']['num_classes']
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for c in range(num_classes):
            tp = np.sum((all_predictions == c) & (all_labels == c))
            fp = np.sum((all_predictions == c) & (all_labels != c))
            fn = np.sum((all_predictions != c) & (all_labels == c))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
        
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': accuracy,
            'precision': np.mean(precision_per_class),
            'recall': np.mean(recall_per_class),
            'f1': np.mean(f1_per_class),
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        return metrics, all_predictions, all_labels
    
    def compute_confusion_matrix(self, predictions, labels):
        """
        Calcule la matrice de confusion.
        
        Args:
            predictions: Prédictions du modèle
            labels: Labels réels
            
        Returns:
            Matrice de confusion numpy
        """
        num_classes = self.config['dataset']['num_classes']
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        
        for pred, label in zip(predictions, labels):
            cm[label][pred] += 1
        
        return cm
    
    def train(self, train_dataset, val_dataset, epochs: int = None):
        """
        Boucle d'entraînement principale.
        
        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation
            epochs: Nombre d'epochs (utilise config si None)
            
        Returns:
            Historique d'entraînement
        """
        if epochs is None:
            epochs = self.config['training']['epochs']
        
        early_stopping = self.config['training']['early_stopping']
        checkpoint_path = self.config['mindspore']['checkpoint_path']
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        print("🚀 Démarrage de l'entraînement...")
        print(f"📊 Epochs: {epochs}, Batch size: {self.config['training']['batch_size']}")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Entraînement
            train_loss = self.train_epoch(train_dataset)
            
            # Validation
            val_metrics, predictions, labels = self.evaluate(val_dataset)
            
            # Logging
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1']
            })
            
            print(f"Epoch [{epoch + 1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Sauvegarder le meilleur modèle
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                
                # Sauvegarder checkpoint
                ckpt_file = os.path.join(checkpoint_path, "best_model.ckpt")
                ms.save_checkpoint(self.model, ckpt_file)
                print(f"  💾 Nouveau meilleur modèle sauvegardé! Accuracy: {self.best_accuracy:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping['patience']:
                print(f"\n⏹️ Early stopping après {epoch + 1} epochs (patience: {early_stopping['patience']})")
                break
            
            print("-" * 60)
        
        print("=" * 60)
        print(f"✅ Entraînement terminé! Meilleure accuracy: {self.best_accuracy:.4f}")
        
        return self.training_history
    
    def save_metrics(self, output_path: str):
        """Sauvegarde l'historique d'entraînement en JSON."""
        os.makedirs(output_path, exist_ok=True)
        
        metrics_file = os.path.join(output_path, "training_history.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"📊 Métriques sauvegardées: {metrics_file}")
    
    def final_evaluation(self, test_dataset, class_names: list):
        """
        Évaluation finale sur le dataset de test.
        
        Args:
            test_dataset: Dataset de test
            class_names: Noms des classes
            
        Returns:
            Rapport d'évaluation complet
        """
        print("\n🧪 Évaluation finale sur le dataset de test...")
        
        metrics, predictions, labels = self.evaluate(test_dataset)
        confusion_matrix = self.compute_confusion_matrix(predictions, labels)
        
        print("\n📈 Résultats finaux:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        
        print("\n📊 Métriques par classe:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"    Recall: {metrics['recall_per_class'][i]:.4f}")
            print(f"    F1: {metrics['f1_per_class'][i]:.4f}")
        
        print("\n🔢 Matrice de confusion:")
        print(confusion_matrix)
        
        return {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix.tolist(),
            'class_names': class_names
        }


if __name__ == "__main__":
    print("🧪 Test du module d'entraînement...")
    
    # Ce module nécessite le modèle et les données
    print("✅ Module d'entraînement chargé avec succès")
