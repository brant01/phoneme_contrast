#!/usr/bin/env python3
"""
Complete Multirun Analysis Pipeline
Analyzes multirun results, finds best model, extracts embeddings, and generates all visualizations.
Usage: python scripts/analyze_multirun_complete.py multirun/2025-05-23/07-21-08/
"""

import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torchaudio
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CompleteMultirunAnalyzer:
    """Complete analysis pipeline for multirun experiments."""
    
    def __init__(self, multirun_dir: Path, output_dir: Optional[Path] = None):
        self.multirun_dir = Path(multirun_dir).resolve()
        self.output_dir = output_dir or self.multirun_dir / 'complete_analysis'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.setup_logging()
        self.logger.info(f"Starting complete analysis of: {self.multirun_dir}")
        
        self.runs_data = []
        self.best_run = None
        self.embeddings = None
        self.labels = None
        self.phonemes = []
        self.label_map = {}
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        self.logger.info("="*60)
        self.logger.info("COMPLETE MULTIRUN ANALYSIS PIPELINE")
        self.logger.info("="*60)
        
        # Step 1: Analyze all runs
        self.analyze_all_runs()
        
        # Step 2: Find best run
        self.find_best_run()
        
        # Step 3: Extract embeddings from best model
        self.extract_embeddings_from_best()
        
        # Step 4: Generate all visualizations
        self.generate_visualizations()
        
        # Step 5: Create final report
        self.create_final_report()
        
        self.logger.info("\n*** Analysis complete! All results saved to: %s", self.output_dir)
        
    def analyze_all_runs(self):
        """Analyze all runs in the multirun directory."""
        self.logger.info("\n[STEP 1] Analyzing all runs...")
        
        # Find all run directories
        run_dirs = [d for d in self.multirun_dir.iterdir() 
                   if d.is_dir() and d.name.isdigit()]
        
        if not run_dirs:
            # Check if we're one level too deep
            parent_runs = [d for d in self.multirun_dir.parent.iterdir() 
                          if d.is_dir() and d.name.isdigit()]
            if parent_runs:
                self.multirun_dir = self.multirun_dir.parent
                run_dirs = parent_runs
        
        self.logger.info(f"Found {len(run_dirs)} runs")
        
        for run_dir in sorted(run_dirs):
            self.logger.info(f"Analyzing run {run_dir.name}...")
            run_data = self.analyze_single_run(run_dir)
            if run_data:
                self.runs_data.append(run_data)
                
        # Create comparison DataFrame
        self.comparison_df = pd.DataFrame([
            {
                'run_id': r['run_id'],
                'temperature': r.get('temperature', np.nan),
                'learning_rate': r.get('learning_rate', np.nan),
                'final_loss': r.get('final_train_loss', np.nan),
                'best_rf_acc': r.get('best_rf_acc', np.nan),
                'best_linear_acc': r.get('best_linear_acc', np.nan)
            }
            for r in self.runs_data
        ])
        
        # Save comparison
        self.comparison_df.to_csv(self.output_dir / 'runs_comparison.csv', index=False)
        self.logger.info("\nRuns summary:")
        self.logger.info(self.comparison_df.to_string())
        
    def analyze_single_run(self, run_dir: Path) -> Dict:
        """Analyze a single run."""
        data = {
            'path': str(run_dir),
            'run_id': run_dir.name
        }
        
        # Load config
        config_path = run_dir / '.hydra' / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                data['config'] = config
                data['temperature'] = config['training']['loss']['temperature']
                data['learning_rate'] = config['training']['learning_rate']
                data['epochs'] = config['training']['epochs']
        
        # Load metrics
        metrics_path = run_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                data['metrics'] = metrics
                
                if 'train_loss' in metrics and metrics['train_loss']:
                    data['final_train_loss'] = metrics['train_loss'][-1]
                    data['best_train_loss'] = min(metrics['train_loss'])
                if 'val_rf_accuracy' in metrics and metrics['val_rf_accuracy']:
                    data['final_rf_acc'] = metrics['val_rf_accuracy'][-1]
                    data['best_rf_acc'] = max(metrics['val_rf_accuracy'])
                if 'val_linear_accuracy' in metrics and metrics['val_linear_accuracy']:
                    data['final_linear_acc'] = metrics['val_linear_accuracy'][-1]
                    data['best_linear_acc'] = max(metrics['val_linear_accuracy'])
        
        # Find checkpoints
        checkpoints = list((run_dir / 'checkpoints').glob('*.pt'))
        data['checkpoints'] = [str(c) for c in checkpoints]
        
        return data
        
    def find_best_run(self):
        """Find the best run based on RF accuracy."""
        self.logger.info("\n[STEP 2] Finding best run...")
        
        if self.comparison_df.empty:
            self.logger.error("No valid runs found!")
            return
            
        # Find best by RF accuracy
        best_idx = self.comparison_df['best_rf_acc'].idxmax()
        self.best_run = self.runs_data[best_idx]
        
        self.logger.info(f"Best run: {self.best_run['run_id']}")
        self.logger.info(f"  Temperature: {self.best_run.get('temperature')}")
        self.logger.info(f"  Learning rate: {self.best_run.get('learning_rate')}")
        self.logger.info(f"  Best RF Accuracy: {self.best_run.get('best_rf_acc', 0):.3f}")
        self.logger.info(f"  Best Linear Accuracy: {self.best_run.get('best_linear_acc', 0):.3f}")
        
    def extract_embeddings_from_best(self):
        """Extract embeddings from the best model."""
        self.logger.info("\n[STEP 3] Extracting embeddings from best model...")
        
        if not self.best_run:
            self.logger.error("No best run found!")
            return
            
        # Find best checkpoint
        checkpoint_path = None
        run_path = Path(self.best_run['path'])
        
        for checkpoint_name in ['checkpoint_best.pt', 'checkpoint_final.pt', 'checkpoint_periodic.pt']:
            possible_path = run_path / 'checkpoints' / checkpoint_name
            if possible_path.exists():
                checkpoint_path = possible_path
                break
                
        if not checkpoint_path:
            self.logger.error("No checkpoint found!")
            return
            
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return
        
        # Always extract embeddings from full dataset for visualization
        self.logger.info("Extracting embeddings from full dataset for visualization...")
        
        try:
            # Import necessary modules
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            
            from src.models.registry import model_registry
            from src.models.phoneme_cnn import PhonemeNet
            from src.datasets.parser import parse_dataset
            from src.utils.logging import create_logger
            
            # Get config from checkpoint or best run
            config = checkpoint.get('config', self.best_run.get('config', {}))
            
            # Create model
            model_config = config.get('model', {})
            model = PhonemeNet(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load ENTIRE dataset (no train/val split)
            data_path = Path(config.get('data', {}).get('data_path', 'data/raw/New Stimuli 9-8-2024'))
            
            # Parse dataset to get ALL phoneme files
            file_paths, labels, label_map, metadata = parse_dataset(data_path, self.logger)
            
            # Create simple data loading for ALL samples
            all_embeddings = []
            all_labels = []
            all_phonemes = []
            
            # Process each file
            self.logger.info(f"Processing {len(file_paths)} audio files...")
            
            with torch.no_grad():
                for i, (file_path, label) in enumerate(zip(file_paths, labels)):
                    if i % 10 == 0:
                        self.logger.info(f"  Processing file {i+1}/{len(file_paths)}")
                    
                    # Load audio
                    waveform, sample_rate = torchaudio.load(file_path)
                    
                    # Simple MFCC extraction
                    target_sr = 16000
                    if sample_rate != target_sr:
                        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                        waveform = resampler(waveform)
                    
                    # Extract MFCC features
                    mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=target_sr,
                        n_mfcc=40,
                        melkwargs={"n_fft": 400, "hop_length": 160}
                    )
                    
                    features = mfcc_transform(waveform)
                    
                    # Ensure mono
                    if features.shape[0] > 1:
                        features = features.mean(dim=0, keepdim=True)
                    
                    # Pad/trim to fixed length
                    max_length = 200  # ~2 seconds
                    if features.shape[-1] > max_length:
                        features = features[..., :max_length]
                    else:
                        pad_length = max_length - features.shape[-1]
                        features = torch.nn.functional.pad(features, (0, pad_length))
                    
                    # Add batch dimension
                    features = features.unsqueeze(0)
                    
                    # Get embedding
                    embedding = model(features)
                    
                    all_embeddings.append(embedding.cpu().numpy())
                    all_labels.append(label)
                    
                    # Get phoneme name from label_map
                    phoneme_name = [k for k, v in label_map.items() if v == label][0]
                    all_phonemes.append(phoneme_name)
            
            self.embeddings = np.vstack(all_embeddings)
            self.labels = np.array(all_labels)
            self.phonemes = all_phonemes
            self.label_map = label_map
            
            self.logger.info(f"Extracted embeddings from full dataset: {self.embeddings.shape}")
            self.logger.info(f"Unique phonemes: {len(set(self.phonemes))}")
            self.logger.info(f"Samples per phoneme: min={np.bincount(self.labels).min()}, max={np.bincount(self.labels).max()}")
            
        except Exception as e:
            self.logger.error(f"Error extracting embeddings: {e}")
            self.logger.info("Falling back to validation embeddings if available...")
            
            # Fallback to saved embeddings if extraction fails
            if 'val_embeddings' in checkpoint:
                self.embeddings = checkpoint['val_embeddings'].numpy()
                self.labels = checkpoint['val_labels'].numpy()
                self.logger.info(f"Using saved validation embeddings: {self.embeddings.shape}")
                self.logger.warning("Note: These are only validation samples, not the full dataset")
            else:
                self.logger.error("No embeddings available!")
                return
            
    def run_inference_for_embeddings(self, checkpoint_path: Path):
        """Run inference to extract embeddings."""
        self.logger.info("Extracting real embeddings from dataset...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {})
            
            # Try to load the model
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            
            # Import necessary modules
            from src.models.registry import model_registry
            from src.models.phoneme_cnn import PhonemeNet
            from src.datasets.parser import parse_dataset
            from src.utils.logging import create_logger
            
            # Create model
            model_config = config.get('model', {})
            model = PhonemeNet(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load dataset
            data_path = Path(config.get('data', {}).get('data_path', 'data/raw/New Stimuli 9-8-2024'))
            
            # Parse dataset to get phoneme names
            file_paths, labels, label_map, metadata = parse_dataset(data_path, self.logger)
            
            # Create simple data loading
            all_embeddings = []
            all_labels = []
            all_phonemes = []
            
            # Process each file
            self.logger.info(f"Processing {len(file_paths)} audio files...")
            
            for i, (file_path, label) in enumerate(zip(file_paths, labels)):
                if i % 10 == 0:
                    self.logger.info(f"  Processing file {i+1}/{len(file_paths)}")
                
                # Load audio
                waveform, sample_rate = torchaudio.load(file_path)
                
                # Simple MFCC extraction
                target_sr = 16000
                if sample_rate != target_sr:
                    resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                    waveform = resampler(waveform)
                
                # Extract MFCC features
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=target_sr,
                    n_mfcc=40,
                    melkwargs={"n_fft": 400, "hop_length": 160}
                )
                
                features = mfcc_transform(waveform)
                
                # Ensure mono
                if features.shape[0] > 1:
                    features = features.mean(dim=0, keepdim=True)
                
                # Pad/trim to fixed length
                max_length = 200  # ~2 seconds
                if features.shape[-1] > max_length:
                    features = features[..., :max_length]
                else:
                    pad_length = max_length - features.shape[-1]
                    features = torch.nn.functional.pad(features, (0, pad_length))
                
                # Add batch dimension
                features = features.unsqueeze(0)
                
                # Get embedding
                with torch.no_grad():
                    embedding = model(features)
                
                all_embeddings.append(embedding.numpy())
                all_labels.append(label)
                
                # Get phoneme name from label_map
                phoneme_name = [k for k, v in label_map.items() if v == label][0]
                all_phonemes.append(phoneme_name)
            
            self.embeddings = np.vstack(all_embeddings)
            self.labels = np.array(all_labels)
            self.phonemes = all_phonemes
            self.label_map = label_map
            
            self.logger.info(f"Extracted embeddings: {self.embeddings.shape}")
            self.logger.info(f"Unique phonemes: {len(set(self.phonemes))}")
            
        except Exception as e:
            self.logger.error(f"Error extracting real embeddings: {e}")
            self.logger.info("Falling back to synthetic embeddings...")
            
            # Fallback to synthetic embeddings
            n_samples = 126
            n_classes = 38
            embedding_dim = 128
            
            self.embeddings = []
            self.labels = []
            self.phonemes = []
            
            # Generate phoneme names from your dataset
            phoneme_names = ['ada', 'apa', 'bi', 'bu', 'da', 'de', 'di', 'du', 
                           'ege', 'ete', 'fa', 'fe', 'fi', 'fu', 'ga', 'ge', 
                           'gi', 'gu', 'ibi', 'isi', 'ka', 'ke', 'ki', 'ku', 
                           'pa', 'pe', 'pi', 'pu', 'sa', 'se', 'si', 'su', 
                           'ta', 'te', 'ti', 'tu', 'ubu', 'uku']
            
            samples_per_class = n_samples // n_classes
            for i in range(n_classes):
                center = np.random.randn(embedding_dim)
                center = center / np.linalg.norm(center)
                
                for j in range(samples_per_class):
                    noise = np.random.randn(embedding_dim) * 0.3
                    emb = center + noise
                    emb = emb / np.linalg.norm(emb)
                    self.embeddings.append(emb)
                    self.labels.append(i)
                    self.phonemes.append(phoneme_names[i % len(phoneme_names)])
                    
            self.embeddings = np.array(self.embeddings)
            self.labels = np.array(self.labels)
            
            self.logger.info(f"Generated synthetic embeddings: {self.embeddings.shape}")
        
    def generate_visualizations(self):
        """Generate all visualizations."""
        self.logger.info("\n[STEP 4] Generating visualizations...")
        
        if self.embeddings is None:
            self.logger.error("No embeddings available!")
            return
            
        # 1. Multirun comparison plots
        self.plot_multirun_comparison()
        
        # 2. t-SNE visualization
        self.plot_tsne()
        
        # 3. Confusion matrix
        self.plot_confusion_matrix()
        
        # 4. Distance analysis
        self.analyze_distances()
        
        # 5. Learning curves for best run
        self.plot_learning_curves()
        
    def plot_multirun_comparison(self):
        """Create comparison plots for all runs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multirun Experiment Comparison', fontsize=16)
        
        # Temperature vs Loss
        ax = axes[0, 0]
        scatter = ax.scatter(self.comparison_df['temperature'], 
                           self.comparison_df['final_loss'],
                           c=self.comparison_df['learning_rate'],
                           s=100, alpha=0.7, cmap='viridis')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Final Training Loss')
        ax.set_title('Loss vs Temperature')
        plt.colorbar(scatter, ax=ax, label='Learning Rate')
        
        # Temperature vs RF Accuracy
        ax = axes[0, 1]
        scatter = ax.scatter(self.comparison_df['temperature'],
                           self.comparison_df['best_rf_acc'],
                           c=self.comparison_df['learning_rate'],
                           s=100, alpha=0.7, cmap='viridis')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Best RF Accuracy')
        ax.set_title('RF Accuracy vs Temperature')
        plt.colorbar(scatter, ax=ax, label='Learning Rate')
        
        # Heatmap of results
        ax = axes[1, 0]
        pivot_table = self.comparison_df.pivot_table(
            values='best_rf_acc',
            index='learning_rate',
            columns='temperature',
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.3f', ax=ax, cmap='YlOrRd')
        ax.set_title('RF Accuracy Heatmap')
        
        # Summary table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        summary_df = self.comparison_df[['run_id', 'temperature', 'learning_rate', 'best_rf_acc']].round(3)
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multirun_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_tsne(self):
        """Create t-SNE visualization."""
        self.logger.info("Creating t-SNE visualization...")
        
        # Run t-SNE
        perplexity = min(30, len(self.embeddings) - 1)
        tsne = TSNE(n_components=2, 
                    learning_rate='auto',
                    perplexity=perplexity,
                    metric='cosine',
                    init='pca',
                    random_state=42)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create plot
        plt.figure(figsize=(15, 12))
        
        # Get unique phonemes
        unique_phonemes = sorted(set(self.phonemes))
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_phonemes))))
        if len(unique_phonemes) > 20:
            # Use additional colors for more than 20 phonemes
            colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, len(unique_phonemes) - 20))])
        
        # Create a mapping from phoneme to color
        phoneme_to_color = {phoneme: colors[i % len(colors)] for i, phoneme in enumerate(unique_phonemes)}
        
        # Plot each phoneme
        for phoneme in unique_phonemes:
            mask = [p == phoneme for p in self.phonemes]
            if any(mask):
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[phoneme_to_color[phoneme]], label=phoneme,
                           alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        plt.title('Phoneme Embeddings t-SNE Visualization', fontsize=16)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Add metrics
        try:
            silhouette = silhouette_score(embeddings_2d, self.labels)
            plt.text(0.02, 0.98, f'Silhouette Score: {silhouette:.3f}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass
            
        # Create legend with columns
        ncol = 3 if len(unique_phonemes) > 20 else 2
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=ncol, fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save embeddings with phoneme names
        np.savez(self.output_dir / 'embeddings.npz',
                embeddings=self.embeddings,
                embeddings_2d=embeddings_2d,
                labels=self.labels,
                phonemes=self.phonemes)
 
    def plot_confusion_matrix(self):
        """Generate confusion matrix using adaptive cross-validation."""
        self.logger.info("Creating confusion matrix...")
        
        from sklearn.model_selection import StratifiedKFold, LeaveOneOut
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        # Check minimum samples per class
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        min_samples = counts.min()
        max_samples = counts.max()
        
        self.logger.info(f"Samples per class: min={min_samples}, max={max_samples}")
        
        # Adaptive cross-validation based on minimum samples
        if min_samples == 1:
            # Use Leave-One-Out for very small dataset
            self.logger.info("Using Leave-One-Out cross-validation due to limited samples")
            all_true = []
            all_pred = []
            
            loo = LeaveOneOut()
            for i, (train_idx, test_idx) in enumerate(loo.split(self.embeddings)):
                if i % 10 == 0:
                    self.logger.info(f"  Processing fold {i+1}/{len(self.embeddings)}")
                
                X_train = self.embeddings[train_idx]
                X_test = self.embeddings[test_idx]
                y_train = self.labels[train_idx]
                y_test = self.labels[test_idx]
                
                # Train k-NN with adaptive k
                k = min(5, len(np.unique(y_train)))
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                pred = knn.predict(X_test)
                
                all_true.append(y_test[0])
                all_pred.append(pred[0])
            
            overall_accuracy = (np.array(all_true) == np.array(all_pred)).mean()
            
        else:
            # Use k-fold with adaptive k
            n_splits = min(5, min_samples)  # Can't have more folds than minimum samples
            self.logger.info(f"Using {n_splits}-fold cross-validation")
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            all_true = []
            all_pred = []
            fold_accuracies = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(self.embeddings, self.labels)):
                X_train = self.embeddings[train_idx]
                X_test = self.embeddings[test_idx]
                y_train = self.labels[train_idx]
                y_test = self.labels[test_idx]
                
                # Train k-NN with adaptive k
                k = min(5, len(np.unique(y_train)))
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                predictions = knn.predict(X_test)
                
                # Track accuracy per fold
                fold_acc = (predictions == y_test).mean()
                fold_accuracies.append(fold_acc)
                self.logger.info(f"  Fold {fold+1} accuracy: {fold_acc:.3f}")
                
                all_true.extend(y_test)
                all_pred.extend(predictions)
            
            overall_accuracy = np.mean(fold_accuracies)
            self.logger.info(f"Overall {n_splits}-fold accuracy: {overall_accuracy:.3f} ± {np.std(fold_accuracies):.3f}")
        
        # Get unique phonemes in order
        unique_phonemes = []
        for label in sorted(np.unique(self.labels)):
            idx = np.where(self.labels == label)[0][0]
            unique_phonemes.append(self.phonemes[idx])
        
        # Create confusion matrix from all predictions
        cm = confusion_matrix(all_true, all_pred, labels=sorted(np.unique(self.labels)))
        
        # Also create a normalized version
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Plot
        plt.figure(figsize=(20, 18))
        
        # Create mask for zero values (optional - to make them white)
        mask = cm == 0
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=unique_phonemes,
                    yticklabels=unique_phonemes,
                    cbar_kws={'label': 'Count'},
                    mask=mask, cbar=True,
                    linewidths=0.5, linecolor='gray',
                    square=True)  # Make cells square
        
        plt.title(f'Phoneme Confusion Matrix\n(CV Accuracy: {overall_accuracy:.3f}, Total samples: {len(self.embeddings)})', 
                fontsize=16)
        plt.xlabel('Predicted Phoneme', fontsize=12)
        plt.ylabel('True Phoneme', fontsize=12)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save normalized version
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=unique_phonemes,
                    yticklabels=unique_phonemes,
                    cbar_kws={'label': 'Percentage'},
                    cbar=True,
                    linewidths=0.5, linecolor='gray',
                    square=True,
                    vmin=0, vmax=1)
        
        plt.title(f'Phoneme Confusion Matrix (Normalized)\n(CV Accuracy: {overall_accuracy:.3f})', fontsize=16)
        plt.xlabel('Predicted Phoneme', fontsize=12)
        plt.ylabel('True Phoneme', fontsize=12)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find most confused pairs
        confused_pairs = []
        for i in range(len(unique_phonemes)):
            for j in range(len(unique_phonemes)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true': unique_phonemes[i],
                        'predicted': unique_phonemes[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm_normalized[i, j])
                    })
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        if confused_pairs:
            self.logger.info("Most confused phoneme pairs:")
            for pair in confused_pairs[:10]:
                self.logger.info(f"  {pair['true']} -> {pair['predicted']}: {pair['count']} times ({pair['percentage']:.1%})")
        
        # Save detailed results
        results = {
            'cv_accuracy': float(overall_accuracy),
            'total_samples': len(self.embeddings),
            'cv_method': 'leave-one-out' if min_samples == 1 else f'{n_splits}-fold',
            'phoneme_order': unique_phonemes,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'samples_per_phoneme': {phoneme: int(count) for phoneme, count in zip(unique_phonemes, counts)},  # Convert to int
            'min_samples_per_class': int(min_samples),
            'max_samples_per_class': int(max_samples)
        }
        
        with open(self.output_dir / 'confusion_matrix_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
    def analyze_distances(self):
        """Analyze embedding distances."""
        self.logger.info("Analyzing embedding distances...")
        
        # Compute pairwise distances
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(self.embeddings)
        
        # Separate intra and inter class distances
        intra_distances = []
        inter_distances = []
        
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                if self.labels[i] == self.labels[j]:
                    intra_distances.append(distances[i, j])
                else:
                    inter_distances.append(distances[i, j])
        
        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.hist(intra_distances, bins=50, alpha=0.5, label='Intra-class', density=True)
        plt.hist(inter_distances, bins=50, alpha=0.5, label='Inter-class', density=True)
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.title('Distribution of Embedding Distances')
        plt.legend()
        
        # Add statistics
        stats_text = f'Intra-class: {np.mean(intra_distances):.3f} ± {np.std(intra_distances):.3f}\n'
        stats_text += f'Inter-class: {np.mean(inter_distances):.3f} ± {np.std(inter_distances):.3f}\n'
        stats_text += f'Separation ratio: {np.mean(inter_distances)/np.mean(intra_distances):.3f}'
        plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distance_distributions.png', dpi=300)
        plt.close()
        
    def plot_learning_curves(self):
        """Plot learning curves for the best run."""
        if not self.best_run or 'metrics' not in self.best_run:
            return
            
        metrics = self.best_run['metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Learning Curves - Best Run (ID: {self.best_run["run_id"]})', fontsize=16)
        
        # Training loss
        if 'train_loss' in metrics:
            ax = axes[0, 0]
            ax.plot(metrics['train_loss'], label='Train Loss')
            if 'val_loss' in metrics:
                val_epochs = np.linspace(0, len(metrics['train_loss'])-1, len(metrics['val_loss']))
                ax.plot(val_epochs, metrics['val_loss'], 'o-', label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        if 'val_rf_accuracy' in metrics:
            ax = axes[0, 1]
            val_epochs = np.linspace(0, len(metrics['train_loss'])-1, len(metrics['val_rf_accuracy']))
            ax.plot(val_epochs, metrics['val_rf_accuracy'], 'o-', label='RF Accuracy')
            if 'val_linear_accuracy' in metrics:
                ax.plot(val_epochs, metrics['val_linear_accuracy'], 's-', label='Linear Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in metrics:
            ax = axes[1, 0]
            ax.plot(metrics['lr'])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Configuration summary
        ax = axes[1, 1]
        ax.axis('off')
        config_text = f"Best Run Configuration:\n\n"
        config_text += f"Temperature: {self.best_run.get('temperature', 'N/A')}\n"
        config_text += f"Learning Rate: {self.best_run.get('learning_rate', 'N/A')}\n"
        config_text += f"Epochs: {self.best_run.get('epochs', 'N/A')}\n"
        config_text += f"Final Loss: {self.best_run.get('final_train_loss', 'N/A'):.3f}\n"
        config_text += f"Best RF Acc: {self.best_run.get('best_rf_acc', 'N/A'):.3f}\n"
        config_text += f"Best Linear Acc: {self.best_run.get('best_linear_acc', 'N/A'):.3f}\n"
        
        ax.text(0.1, 0.9, config_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves_best.png', dpi=300)
        plt.close()
        
    def create_final_report(self):
        """Create a comprehensive final report."""
        self.logger.info("\n[STEP 5] Creating final report...")
        
        # Get values with proper formatting
        best_run_id = self.best_run['run_id'] if self.best_run else 'N/A'
        temperature = self.best_run.get('temperature', 'N/A') if self.best_run else 'N/A'
        learning_rate = self.best_run.get('learning_rate', 'N/A') if self.best_run else 'N/A'
        
        # Format numeric values only if they exist
        if self.best_run:
            best_rf_acc = f"{self.best_run.get('best_rf_acc', 0):.3f}"
            best_linear_acc = f"{self.best_run.get('best_linear_acc', 0):.3f}"
            final_train_loss = f"{self.best_run.get('final_train_loss', 0):.3f}"
        else:
            best_rf_acc = 'N/A'
            best_linear_acc = 'N/A'
            final_train_loss = 'N/A'
        
        report = f"""# Complete Multirun Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Directory: {self.multirun_dir}

## Summary

Total Runs Analyzed: {len(self.runs_data)}

### Best Run: {best_run_id}
- Temperature: {temperature}
- Learning Rate: {learning_rate}
- Best RF Accuracy: {best_rf_acc}
- Best Linear Accuracy: {best_linear_acc}
- Final Training Loss: {final_train_loss}

## All Runs Comparison

{self.comparison_df.to_string() if hasattr(self, 'comparison_df') else 'No data available'}

## Generated Visualizations

1. **multirun_comparison.png** - Comparison of all runs
2. **tsne_visualization.png** - t-SNE embedding visualization
3. **confusion_matrix.png** - Phoneme confusion matrix
4. **distance_distributions.png** - Intra vs inter-class distances
5. **learning_curves_best.png** - Learning curves for best run

## Files Generated

- runs_comparison.csv - Detailed comparison data
- embeddings.npz - Extracted embeddings
- analysis.log - Complete analysis log
- final_report.md - This report

## Next Steps

1. Review confusion matrix for most confusable phoneme pairs
2. Examine t-SNE clustering for phonetic feature groupings
3. Compare results with human perceptual data
4. Consider fine-tuning on specific phoneme contrasts

---
Analysis complete! All results saved to: {self.output_dir}
"""
        
        # Save report
        report_path = self.output_dir / 'final_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Final report saved to: {report_path}")
        
        # Also save as HTML for better viewing
        try:
            import markdown
            html_content = markdown.markdown(report)
            html_path = self.output_dir / 'final_report.html'
            with open(html_path, 'w') as f:
                f.write(f"<html><body>{html_content}</body></html>")
            self.logger.info(f"HTML report saved to: {html_path}")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='Complete multirun analysis pipeline')
    parser.add_argument('multirun_dir', type=Path, help='Path to multirun directory')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: multirun_dir/complete_analysis)')
    args = parser.parse_args()
    
    # Run analysis
    analyzer = CompleteMultirunAnalyzer(args.multirun_dir, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()