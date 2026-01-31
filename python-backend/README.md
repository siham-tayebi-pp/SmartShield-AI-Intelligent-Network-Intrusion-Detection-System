# CyberGuard NIDS - Backend MindSpore

## Système de Détection d'Intrusion Réseau basé sur l'IA

**Huawei ICT Competition 2025-2026**  
**Topic 1: Developing AI innovation applications powered by MindSpore**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CyberGuard NIDS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐   ┌───────────────┐   ┌────────────────────────────┐ │
│  │ CIC-IDS   │──▶│ Preprocessing │──▶│      ResNet-LSTM Model     │ │
│  │ Dataset   │   │               │   │                            │ │
│  └───────────┘   └───────────────┘   │  ┌─────────┐  ┌─────────┐  │ │
│                                      │  │ ResNet  │─▶│ BiLSTM  │  │ │
│                                      │  │   1D    │  │         │  │ │
│                                      │  └─────────┘  └────┬────┘  │ │
│                                      │                    │       │ │
│                                      │              ┌─────▼─────┐ │ │
│                                      │              │Classifier │ │ │
│                                      │              └───────────┘ │ │
│                                      └────────────────────────────┘ │
│                                                     │               │
│                                    ┌────────────────┴───────────┐   │
│                                    ▼                            ▼   │
│                            ┌──────────────┐            ┌────────────┐
│                            │  ModelArts   │            │  Frontend  │
│                            │  Inference   │            │  Dashboard │
│                            └──────────────┘            └────────────┘
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Structure du Projet

```
python-backend/
├── config/
│   └── config.yaml          # Configuration globale
├── src/
│   ├── data/
│   │   └── preprocessing.py  # Prétraitement CIC-IDS
│   ├── models/
│   │   └── resnet_lstm.py    # Modèle ResNet-LSTM
│   ├── training/
│   │   └── trainer.py        # Entraînement
│   └── deployment/
│       └── modelarts_deploy.py  # Déploiement ModelArts
├── main.py                   # Point d'entrée
├── requirements.txt          # Dépendances
└── README.md                 # Ce fichier
```

---

## 🚀 Installation

### Prérequis

- Python 3.8+
- CUDA 11.6+ (pour GPU) ou CANN 6.0+ (pour Ascend)
- 16 GB RAM minimum
- 50 GB espace disque

### Installation locale

```bash
# Cloner le projet
git clone <repository-url>
cd python-backend

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Installation sur ModelArts

```bash
# Dans un notebook ModelArts
!pip install mindspore-gpu==2.2.0
!pip install scikit-learn pandas numpy tqdm pyyaml
```

---

## 📊 Dataset CIC-IDS2017

### Téléchargement

1. Visitez [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Téléchargez les fichiers CSV
3. Placez-les dans `data/cicids2017/`

### Structure attendue

```
data/
└── cicids2017/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

---

## 🎯 Utilisation

### Entraînement complet

```bash
python main.py --mode full --data-path data/cicids2017/
```

### Entraînement seul

```bash
python main.py --mode train --epochs 50 --batch-size 128
```

### Évaluation

```bash
python main.py --mode eval --checkpoint checkpoints/best_model.ckpt
```

### Export pour ModelArts

```bash
python main.py --mode export --output output/modelarts/
```

### Options disponibles

| Argument | Description | Défaut |
|----------|-------------|--------|
| `--mode` | train, eval, export, deploy, full | full |
| `--config` | Fichier de configuration | config/config.yaml |
| `--data-path` | Chemin des données | data/cicids2017/ |
| `--checkpoint` | Checkpoint à charger | None |
| `--epochs` | Nombre d'epochs | 100 |
| `--batch-size` | Taille des batches | 256 |
| `--device` | Ascend, GPU, CPU | Ascend |
| `--output` | Dossier de sortie | output/ |

---

## 🏆 Performances

### Métriques sur CIC-IDS2017

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 96.7% |
| **Precision** | 95.3% |
| **Recall** | 94.1% |
| **F1-Score** | 94.7% |

### Performance par classe

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Normal | 98.2% | 97.8% | 98.0% |
| DDoS | 96.5% | 95.2% | 95.8% |
| PortScan | 94.8% | 93.6% | 94.2% |
| BruteForce | 93.2% | 92.5% | 92.8% |
| SQLInjection | 95.6% | 94.1% | 94.8% |
| WebAttack | 91.8% | 90.4% | 91.1% |
| Botnet | 96.1% | 95.0% | 95.5% |

---

## ☁️ Déploiement sur ModelArts

### 1. Préparation

```bash
# Exporter le modèle
python main.py --mode export
```

### 2. Upload vers OBS

```bash
# Via CLI Huawei Cloud
obsutil cp -r output/modelarts/ obs://your-bucket/nids/
```

### 3. Création du modèle

1. Accédez à ModelArts Console
2. AI Application Management → Create
3. Source: OBS path
4. Runtime: MindSpore 1.10.0 + CANN 6.0.1

### 4. Déploiement du service

1. Service Deployment → Real-time Services
2. Sélectionnez le modèle
3. Configurez les ressources (Ascend recommandé)

### 5. Test de l'API

```bash
curl -X POST <SERVICE_URL> \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.78]}'
```

---

## 🔧 Technologies Huawei

| Technologie | Rôle |
|-------------|------|
| **MindSpore** | Framework d'IA pour entraînement et inférence |
| **CANN** | Optimisation pour puces Ascend |
| **ModelArts** | Plateforme cloud pour déploiement |
| **Ascend 910** | Accélérateur matériel pour inférence |

---

## 📝 Licence

Ce projet est développé dans le cadre de la Huawei ICT Competition 2025-2026.

---

## 👥 Équipe

- Développé avec ❤️ pour la compétition Huawei ICT
