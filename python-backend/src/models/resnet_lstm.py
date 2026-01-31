"""
Modèle ResNet-LSTM pour la Détection d'Intrusion Réseau
Huawei ICT Competition 2025-2026 - MindSpore

Architecture:
- ResNet 1D pour l'extraction de features des paquets réseau
- LSTM bidirectionnel pour capturer les patterns temporels
- Classificateur dense pour la prédiction multi-classes
"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import HeNormal, Normal
import numpy as np


class ResidualBlock1D(nn.Cell):
    """
    Bloc résiduel 1D adapté pour les données de trafic réseau.
    
    Utilise des convolutions 1D pour traiter les features des paquets
    comme des séquences de caractéristiques.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            stride: Stride de la convolution
        """
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1,
            has_bias=False, weight_init=HeNormal()
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            has_bias=False, weight_init=HeNormal()
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection avec projection si dimensions différentes
        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.SequentialCell([
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, has_bias=False, weight_init=HeNormal()),
                nn.BatchNorm1d(out_channels)
            ])
    
    def construct(self, x):
        """Forward pass du bloc résiduel."""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Cell):
    """
    ResNet 1D pour l'extraction de features des paquets réseau.
    
    Adapté du ResNet classique pour traiter les features numériques
    des paquets comme des données 1D.
    """
    
    def __init__(self, in_channels: int = 1, num_blocks: list = [2, 2, 2, 2],
                 num_features: int = 512):
        """
        Args:
            in_channels: Canaux d'entrée (1 pour features 1D)
            num_blocks: Nombre de blocs par stage
            num_features: Dimension des features de sortie
        """
        super(ResNet1D, self).__init__()
        
        self.in_channels = 64
        
        # Couche d'entrée
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3,
            has_bias=False, weight_init=HeNormal()
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, pad_mode='same')
        
        # Stages ResNet
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Adaptive pooling et projection
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Dense(512, num_features, weight_init=HeNormal())
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        """Crée un stage de blocs résiduels."""
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.SequentialCell(layers)
    
    def construct(self, x):
        """Forward pass du ResNet."""
        # x shape: (batch, 1, num_features)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        return x


class BidirectionalLSTM(nn.Cell):
    """
    LSTM bidirectionnel pour capturer les patterns temporels.
    
    Traite les features extraites par ResNet comme une séquence
    pour détecter les patterns d'attaque sur plusieurs paquets.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            input_size: Dimension des features d'entrée
            hidden_size: Taille cachée du LSTM
            num_layers: Nombre de couches LSTM
            dropout: Taux de dropout
        """
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output size = hidden_size * 2 (bidirectional)
        self.output_size = hidden_size * 2
    
    def construct(self, x):
        """
        Forward pass du LSTM.
        
        Args:
            x: Tensor (batch, features) ou (batch, seq, features)
        """
        # Ajouter dimension séquentielle si nécessaire
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)
        
        # Concaténer les états cachés des deux directions
        # h_n shape: (num_layers * 2, batch, hidden_size)
        forward_h = h_n[-2, :, :]  # Dernier état forward
        backward_h = h_n[-1, :, :]  # Dernier état backward
        
        combined = ops.concat((forward_h, backward_h), axis=1)
        
        return combined


class NIDSClassifier(nn.Cell):
    """
    Classificateur final pour la détection d'intrusion.
    
    Combine les features du ResNet et LSTM pour prédire
    la classe d'attaque (Normal, DDoS, PortScan, etc.)
    """
    
    def __init__(self, input_size: int, num_classes: int = 7,
                 hidden_dim: int = 128, dropout: float = 0.5):
        """
        Args:
            input_size: Dimension des features d'entrée
            num_classes: Nombre de classes d'attaques
            hidden_dim: Dimension de la couche cachée
            dropout: Taux de dropout
        """
        super(NIDSClassifier, self).__init__()
        
        self.classifier = nn.SequentialCell([
            nn.Dense(input_size, hidden_dim, weight_init=HeNormal()),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_dim, hidden_dim // 2, weight_init=HeNormal()),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_dim // 2, num_classes, weight_init=Normal())
        ])
    
    def construct(self, x):
        """Forward pass du classificateur."""
        return self.classifier(x)


class ResNetLSTMNIDS(nn.Cell):
    """
    Modèle complet ResNet-LSTM pour la Détection d'Intrusion Réseau.
    
    Architecture:
    1. ResNet1D: Extrait des features de haut niveau des paquets
    2. BiLSTM: Capture les patterns temporels d'attaque
    3. Classifier: Prédit la classe d'attaque
    
    Ce modèle est optimisé pour:
    - Huawei Ascend 910 via CANN
    - Entraînement sur ModelArts
    - Inférence en temps réel
    """
    
    def __init__(self, num_features: int = 78, num_classes: int = 7, config: dict = None):
        """
        Args:
            num_features: Nombre de features des paquets (78 pour CIC-IDS)
            num_classes: Nombre de classes d'attaques
            config: Configuration optionnelle
        """
        super(ResNetLSTMNIDS, self).__init__()
        
        # Configuration par défaut
        if config is None:
            config = {
                'resnet': {'num_blocks': [2, 2, 2, 2], 'num_features': 512},
                'lstm': {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
                'classifier': {'hidden_dim': 128, 'dropout': 0.5}
            }
        
        # ResNet pour extraction de features
        self.resnet = ResNet1D(
            in_channels=1,
            num_blocks=config['resnet']['num_blocks'],
            num_features=config['resnet']['num_features']
        )
        
        # LSTM bidirectionnel
        self.lstm = BidirectionalLSTM(
            input_size=config['resnet']['num_features'],
            hidden_size=config['lstm']['hidden_size'],
            num_layers=config['lstm']['num_layers'],
            dropout=config['lstm']['dropout']
        )
        
        # Classificateur
        lstm_output_size = config['lstm']['hidden_size'] * 2  # Bidirectional
        self.classifier = NIDSClassifier(
            input_size=lstm_output_size,
            num_classes=num_classes,
            hidden_dim=config['classifier']['hidden_dim'],
            dropout=config['classifier']['dropout']
        )
        
        self.num_features = num_features
        self.num_classes = num_classes
    
    def construct(self, x):
        """
        Forward pass complet du modèle.
        
        Args:
            x: Tensor de shape (batch, 1, num_features)
            
        Returns:
            Logits de shape (batch, num_classes)
        """
        # ResNet: extraction de features
        features = self.resnet(x)
        
        # LSTM: patterns temporels
        temporal_features = self.lstm(features)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        return logits
    
    def predict(self, x):
        """
        Prédit les classes avec probabilités.
        
        Args:
            x: Tensor d'entrée
            
        Returns:
            Tuple (classes prédites, probabilités)
        """
        logits = self.construct(x)
        softmax = ops.Softmax(axis=1)
        probs = softmax(logits)
        predictions = ops.Argmax(axis=1)(logits)
        
        return predictions, probs


def create_model(config: dict = None, num_features: int = 78, 
                 num_classes: int = 7) -> ResNetLSTMNIDS:
    """
    Factory function pour créer le modèle NIDS.
    
    Args:
        config: Configuration du modèle
        num_features: Nombre de features
        num_classes: Nombre de classes
        
    Returns:
        Instance du modèle ResNetLSTMNIDS
    """
    model = ResNetLSTMNIDS(
        num_features=num_features,
        num_classes=num_classes,
        config=config
    )
    
    return model


if __name__ == "__main__":
    # Test du modèle
    print("🧪 Test du modèle ResNet-LSTM NIDS...")
    
    # Créer le modèle
    model = create_model(num_features=78, num_classes=7)
    
    # Données de test
    batch_size = 4
    num_features = 78
    
    # Simuler un batch d'entrée
    test_input = Tensor(np.random.randn(batch_size, 1, num_features).astype(np.float32))
    
    # Forward pass
    output = model(test_input)
    
    print(f"✅ Input shape: {test_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Nombre de paramètres: {sum(p.size for p in model.get_parameters())}")
    
    # Test prédiction
    predictions, probs = model.predict(test_input)
    print(f"✅ Predictions: {predictions}")
    print(f"✅ Probabilités max: {probs.max(axis=1)}")
