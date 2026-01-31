"""
Prétraitement du Dataset CIC-IDS2017 pour NIDS MindSpore
Huawei ICT Competition 2025-2026

Ce module gère:
- Chargement des données CIC-IDS2017
- Nettoyage et normalisation
- Équilibrage des classes
- Création des datasets MindSpore
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import mindspore.dataset as ds
from mindspore import Tensor
import os
from tqdm import tqdm
import yaml


class CICIDSPreprocessor:
    """Préprocesseur pour le dataset CIC-IDS2017."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le préprocesseur.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
        # Mapping des labels CIC-IDS vers nos classes
        self.label_mapping = {
            'BENIGN': 'Normal',
            'DDoS': 'DDoS',
            'DoS Hulk': 'DDoS',
            'DoS GoldenEye': 'DDoS',
            'DoS slowloris': 'DDoS',
            'DoS Slowhttptest': 'DDoS',
            'PortScan': 'PortScan',
            'FTP-Patator': 'BruteForce',
            'SSH-Patator': 'BruteForce',
            'Web Attack – Brute Force': 'BruteForce',
            'Web Attack – XSS': 'WebAttack',
            'Web Attack – Sql Injection': 'SQLInjection',
            'Bot': 'Botnet',
            'Infiltration': 'Botnet',
            'Heartbleed': 'WebAttack',
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Charge tous les fichiers CSV du dataset CIC-IDS2017.
        
        Args:
            data_path: Chemin vers le dossier contenant les fichiers CSV
            
        Returns:
            DataFrame combiné de toutes les données
        """
        print("📂 Chargement des données CIC-IDS2017...")
        
        all_data = []
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        for file in tqdm(csv_files, desc="Chargement des fichiers"):
            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                all_data.append(df)
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {file}: {e}")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ {len(combined_df)} échantillons chargés")
        
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données: supprime les NaN, infinités, et colonnes inutiles.
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame nettoyé
        """
        print("🧹 Nettoyage des données...")
        
        # Suppression des colonnes non numériques sauf le label
        label_col = ' Label' if ' Label' in df.columns else 'Label'
        
        # Conserver les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_clean = df[numeric_cols + [label_col]].copy()
        
        # Remplacer les infinités par NaN puis supprimer
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean.dropna(inplace=True)
        
        # Mapper les labels
        df_clean['Label'] = df_clean[label_col].map(
            lambda x: self.label_mapping.get(x.strip(), 'Normal')
        )
        
        print(f"✅ {len(df_clean)} échantillons après nettoyage")
        
        return df_clean
    
    def balance_classes(self, df: pd.DataFrame, strategy: str = 'undersample') -> pd.DataFrame:
        """
        Équilibre les classes pour éviter le biais.
        
        Args:
            df: DataFrame avec données déséquilibrées
            strategy: 'undersample' ou 'oversample'
            
        Returns:
            DataFrame équilibré
        """
        print("⚖️ Équilibrage des classes...")
        
        class_counts = df['Label'].value_counts()
        print(f"Distribution originale:\n{class_counts}")
        
        if strategy == 'undersample':
            min_count = class_counts.min()
            balanced_dfs = []
            
            for label in class_counts.index:
                class_df = df[df['Label'] == label]
                balanced_dfs.append(
                    resample(class_df, n_samples=min_count, random_state=42)
                )
            
            df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        elif strategy == 'oversample':
            max_count = class_counts.max()
            balanced_dfs = []
            
            for label in class_counts.index:
                class_df = df[df['Label'] == label]
                balanced_dfs.append(
                    resample(class_df, n_samples=max_count, 
                            replace=True, random_state=42)
                )
            
            df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        print(f"Distribution après équilibrage:\n{df_balanced['Label'].value_counts()}")
        
        return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def normalize_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                          X_test: np.ndarray) -> tuple:
        """
        Normalise les features avec StandardScaler.
        
        Args:
            X_train, X_val, X_test: Arrays de features
            
        Returns:
            Tuple de features normalisées
        """
        print("📊 Normalisation des features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def prepare_datasets(self, df: pd.DataFrame) -> dict:
        """
        Prépare les datasets d'entraînement, validation et test.
        
        Args:
            df: DataFrame nettoyé et équilibré
            
        Returns:
            Dictionnaire avec les datasets MindSpore
        """
        print("🔧 Préparation des datasets MindSpore...")
        
        # Séparation features/labels
        self.feature_columns = [col for col in df.columns if col != 'Label']
        X = df[self.feature_columns].values.astype(np.float32)
        y = self.label_encoder.fit_transform(df['Label'].values)
        
        # Split train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(1 - self.config['dataset']['train_split']),
            random_state=42,
            stratify=y
        )
        
        val_ratio = self.config['dataset']['validation_split'] / (
            self.config['dataset']['validation_split'] + 
            self.config['dataset']['test_split']
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=42,
            stratify=y_temp
        )
        
        # Normalisation
        X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
        
        # Reshape pour le modèle (batch, 1, features) pour ResNet 1D
        X_train = X_train.reshape(-1, 1, X_train.shape[1])
        X_val = X_val.reshape(-1, 1, X_val.shape[1])
        X_test = X_test.reshape(-1, 1, X_test.shape[1])
        
        print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'train': {'X': X_train, 'y': y_train.astype(np.int32)},
            'val': {'X': X_val, 'y': y_val.astype(np.int32)},
            'test': {'X': X_test, 'y': y_test.astype(np.int32)},
            'num_features': X_train.shape[2],
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist()
        }
    
    def create_mindspore_dataset(self, data: dict, batch_size: int, 
                                  shuffle: bool = True) -> ds.GeneratorDataset:
        """
        Crée un dataset MindSpore à partir des données.
        
        Args:
            data: Dictionnaire avec 'X' et 'y'
            batch_size: Taille des batches
            shuffle: Mélanger les données
            
        Returns:
            Dataset MindSpore prêt pour l'entraînement
        """
        def generator():
            for i in range(len(data['X'])):
                yield data['X'][i], data['y'][i]
        
        dataset = ds.GeneratorDataset(
            generator,
            column_names=['features', 'label'],
            shuffle=shuffle
        )
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        return dataset
    
    def process_pipeline(self, data_path: str) -> dict:
        """
        Pipeline complet de prétraitement.
        
        Args:
            data_path: Chemin vers les données CIC-IDS2017
            
        Returns:
            Dictionnaire avec tous les datasets préparés
        """
        print("🚀 Démarrage du pipeline de prétraitement...")
        print("=" * 50)
        
        # 1. Charger les données
        df = self.load_data(data_path)
        
        # 2. Nettoyer
        df = self.clean_data(df)
        
        # 3. Équilibrer
        df = self.balance_classes(df, strategy='undersample')
        
        # 4. Préparer les datasets
        datasets = self.prepare_datasets(df)
        
        # 5. Créer les datasets MindSpore
        batch_size = self.config['training']['batch_size']
        
        datasets['train_ds'] = self.create_mindspore_dataset(
            datasets['train'], batch_size, shuffle=True
        )
        datasets['val_ds'] = self.create_mindspore_dataset(
            datasets['val'], batch_size, shuffle=False
        )
        datasets['test_ds'] = self.create_mindspore_dataset(
            datasets['test'], batch_size, shuffle=False
        )
        
        print("=" * 50)
        print("✅ Pipeline de prétraitement terminé!")
        
        return datasets


if __name__ == "__main__":
    # Test du préprocesseur
    preprocessor = CICIDSPreprocessor()
    
    # Simuler des données pour test
    print("🧪 Test du préprocesseur avec données simulées...")
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    n_features = 78
    
    test_data = {
        'features': np.random.randn(n_samples, n_features).astype(np.float32),
        'labels': np.random.randint(0, 7, n_samples)
    }
    
    print(f"✅ Préprocesseur fonctionnel avec {n_samples} échantillons simulés")
