#!/usr/bin/env python3
"""
CyberGuard NIDS - Système de Détection d'Intrusion basé sur MindSpore
Huawei ICT Competition 2025-2026

Point d'entrée principal pour:
- Prétraitement des données CIC-IDS2017
- Entraînement du modèle ResNet-LSTM
- Évaluation et export des métriques
- Déploiement sur ModelArts
"""

import argparse
import os
import sys
import yaml
import json
from datetime import datetime

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessing import CICIDSPreprocessor
from src.models.resnet_lstm import create_model
from src.training.trainer import NIDSTrainer


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="CyberGuard NIDS - MindSpore Training Pipeline"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "export", "deploy", "full"],
        default="full",
        help="Mode d'exécution (train, eval, export, deploy, full)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/cicids2017/",
        help="Chemin vers les données CIC-IDS2017"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin vers un checkpoint existant"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Nombre d'epochs (override config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Taille des batches (override config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["Ascend", "GPU", "CPU"],
        default=None,
        help="Device cible (override config)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Dossier de sortie"
    )
    
    return parser.parse_args()


def print_banner():
    """Affiche la bannière du projet."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗ ██╗   ██╗ █████╗  ║
║  ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗ ║
║  ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝██║  ███╗██║   ██║███████║ ║
║  ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗██║   ██║██║   ██║██╔══██║ ║
║  ╚██████╗   ██║   ██████╔╝███████╗██║  ██║╚██████╔╝╚██████╔╝██║  ██║ ║
║   ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ║
║                                                                      ║
║            Network Intrusion Detection System                        ║
║            Powered by MindSpore & Huawei Ascend                      ║
║                                                                      ║
║            Huawei ICT Competition 2025-2026                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_training(args, config):
    """Exécute le pipeline d'entraînement."""
    print("\n" + "=" * 60)
    print("📚 PHASE 1: PRÉTRAITEMENT DES DONNÉES")
    print("=" * 60)
    
    # Prétraitement
    preprocessor = CICIDSPreprocessor(args.config)
    
    if os.path.exists(args.data_path):
        datasets = preprocessor.process_pipeline(args.data_path)
    else:
        print(f"⚠️ Données non trouvées: {args.data_path}")
        print("📝 Utilisation de données simulées pour la démonstration...")
        
        # Données simulées pour démo
        import numpy as np
        n_train, n_val, n_test = 8000, 1000, 1000
        num_features = 78
        num_classes = 7
        
        datasets = {
            'train': {
                'X': np.random.randn(n_train, 1, num_features).astype(np.float32),
                'y': np.random.randint(0, num_classes, n_train).astype(np.int32)
            },
            'val': {
                'X': np.random.randn(n_val, 1, num_features).astype(np.float32),
                'y': np.random.randint(0, num_classes, n_val).astype(np.int32)
            },
            'test': {
                'X': np.random.randn(n_test, 1, num_features).astype(np.float32),
                'y': np.random.randint(0, num_classes, n_test).astype(np.int32)
            },
            'num_features': num_features,
            'num_classes': num_classes,
            'class_names': ["Normal", "DDoS", "PortScan", "BruteForce", 
                           "SQLInjection", "WebAttack", "Botnet"]
        }
        
        # Créer les datasets MindSpore
        batch_size = args.batch_size or config['training']['batch_size']
        datasets['train_ds'] = preprocessor.create_mindspore_dataset(
            datasets['train'], batch_size, shuffle=True
        )
        datasets['val_ds'] = preprocessor.create_mindspore_dataset(
            datasets['val'], batch_size, shuffle=False
        )
        datasets['test_ds'] = preprocessor.create_mindspore_dataset(
            datasets['test'], batch_size, shuffle=False
        )
    
    print("\n" + "=" * 60)
    print("🏗️ PHASE 2: CRÉATION DU MODÈLE")
    print("=" * 60)
    
    # Créer le modèle
    model = create_model(
        num_features=datasets['num_features'],
        num_classes=datasets['num_classes'],
        config=config.get('model', {}).get('architecture')
    )
    
    # Charger checkpoint si fourni
    if args.checkpoint and os.path.exists(args.checkpoint):
        import mindspore as ms
        ms.load_checkpoint(args.checkpoint, model)
        print(f"✅ Checkpoint chargé: {args.checkpoint}")
    
    print("\n" + "=" * 60)
    print("🎯 PHASE 3: ENTRAÎNEMENT")
    print("=" * 60)
    
    # Entraînement
    trainer = NIDSTrainer(model, args.config)
    
    epochs = args.epochs or config['training']['epochs']
    history = trainer.train(
        datasets['train_ds'],
        datasets['val_ds'],
        epochs=epochs
    )
    
    # Sauvegarder l'historique
    trainer.save_metrics(args.output)
    
    print("\n" + "=" * 60)
    print("🧪 PHASE 4: ÉVALUATION FINALE")
    print("=" * 60)
    
    # Évaluation finale
    eval_results = trainer.final_evaluation(
        datasets['test_ds'],
        datasets['class_names']
    )
    
    # Sauvegarder les résultats
    results_file = os.path.join(args.output, "evaluation_results.json")
    with open(results_file, 'w') as f:
        # Convertir les arrays numpy en listes pour JSON
        eval_results['metrics'] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in eval_results['metrics'].items()
        }
        json.dump(eval_results, f, indent=2)
    
    print(f"\n📊 Résultats sauvegardés: {results_file}")
    
    return model, datasets


def run_export(args, config, model=None):
    """Exporte le modèle pour ModelArts."""
    print("\n" + "=" * 60)
    print("📦 EXPORT DU MODÈLE")
    print("=" * 60)
    
    from src.deployment.modelarts_deploy import ModelArtsDeployer
    
    deployer = ModelArtsDeployer(args.config)
    
    if model is None:
        model = create_model(num_features=78, num_classes=7)
    
    checkpoint_path = args.checkpoint or "checkpoints/best_model.ckpt"
    output_path = os.path.join(args.output, "modelarts/")
    
    class_names = ["Normal", "DDoS", "PortScan", "BruteForce", 
                   "SQLInjection", "WebAttack", "Botnet"]
    
    if os.path.exists(checkpoint_path):
        deployer.export_model(model, checkpoint_path, output_path)
    else:
        print("⚠️ Pas de checkpoint trouvé, création des fichiers de config seulement")
        os.makedirs(output_path, exist_ok=True)
    
    deployer.create_model_config(output_path, class_names)
    deployer.create_inference_code(output_path)
    
    print(f"\n✅ Fichiers exportés vers: {output_path}")


def run_deploy(args, config):
    """Déploie le modèle sur ModelArts."""
    print("\n" + "=" * 60)
    print("🚀 DÉPLOIEMENT SUR MODELARTS")
    print("=" * 60)
    
    from src.deployment.modelarts_deploy import ModelArtsDeployer
    
    deployer = ModelArtsDeployer(args.config)
    
    output_path = os.path.join(args.output, "modelarts/")
    deployer.deploy_to_modelarts(output_path + "/nids_model.air", output_path)


def main():
    """Point d'entrée principal."""
    print_banner()
    
    args = parse_args()
    
    # Charger la configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device si spécifié
    if args.device:
        config['mindspore']['device_target'] = args.device
    
    # Override batch size si spécifié
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Créer le dossier de sortie
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n📁 Configuration: {args.config}")
    print(f"📁 Données: {args.data_path}")
    print(f"📁 Sortie: {args.output}")
    print(f"🖥️ Device: {config['mindspore']['device_target']}")
    print(f"📊 Mode: {args.mode}")
    
    model = None
    datasets = None
    
    try:
        if args.mode in ["train", "full"]:
            model, datasets = run_training(args, config)
        
        if args.mode in ["export", "full"]:
            run_export(args, config, model)
        
        if args.mode in ["deploy", "full"]:
            run_deploy(args, config)
        
        if args.mode == "eval":
            if args.checkpoint:
                print("⚠️ Mode eval requiert --checkpoint")
            else:
                model, datasets = run_training(args, config)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
