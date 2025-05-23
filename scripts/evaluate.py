# scripts/evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import json

from src.datasets.parser import parse_dataset
from src.datasets.dataset import PhonemeContrastiveDataset
from src.datasets.features import build_feature_extractor
from src.models import model_registry
from src.utils.device import get_best_device
from src.utils.logging import create_logger


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract embeddings from the model."""
    model.eval()
    embeddings = []
    labels = []
    metadata = []
    
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        views = batch['views'].to(device)
        batch_labels = batch['label']
        batch_metadata = batch['metadata']
        
        # Handle single view for evaluation
        if views.dim() == 4:  # [batch, C, H, W]
            emb = model(views)
        else:  # [batch, n_views, C, H, W]
            # Use first view only
            emb = model(views[:, 0])
            
        embeddings.append(emb.cpu())
        labels.extend(batch_labels.tolist())
        metadata.extend(batch_metadata)
        
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = np.array(labels)
    
    return embeddings, labels, metadata


def plot_confusion_matrix(embeddings, labels, label_map, output_dir):
    """Generate confusion matrix using k-NN classifier."""
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embeddings, labels)
    predictions = knn.predict(embeddings)
    
    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Get phoneme names
    phoneme_names = sorted(label_map.keys(), key=lambda x: label_map[x])
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=phoneme_names,
                yticklabels=phoneme_names)
    plt.title('Phoneme Confusion Matrix (k-NN)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    # Save classification report
    report = classification_report(labels, predictions, 
                                  target_names=phoneme_names,
                                  output_dict=True)
    
    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    return cm, report


def plot_tsne(embeddings, labels, metadata, label_map, output_dir):
    """Create t-SNE visualization of embeddings."""
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Colored by phoneme
    scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=labels, cmap='tab20', alpha=0.7)
    axes[0].set_title('t-SNE: Phoneme Clusters')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Add phoneme labels to cluster centers
    for label in np.unique(labels):
        mask = labels == label
        center = embeddings_2d[mask].mean(axis=0)
        phoneme = [k for k, v in label_map.items() if v == label][0]
        axes[0].annotate(phoneme, center, fontsize=8, weight='bold')
    
    # Plot 2: Colored by gender
    gender_map = {'male': 0, 'female': 1, 'unknown': 2}
    gender_colors = [gender_map[m['gender']] for m in metadata]
    
    scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=gender_colors, cmap='viridis', alpha=0.7)
    axes[1].set_title('t-SNE: Speaker Gender')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='purple', label='Male'),
                      Patch(facecolor='yellow', label='Female')]
    axes[1].legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_visualization.png', dpi=300)
    plt.close()
    
    return embeddings_2d


def analyze_distances(embeddings, labels, label_map, output_dir):
    """Analyze inter and intra-class distances."""
    n_samples = len(embeddings)
    distances = np.zeros((n_samples, n_samples))
    
    # Compute pairwise distances
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    
    # Compute statistics
    results = {}
    
    for label in np.unique(labels):
        mask = labels == label
        phoneme = [k for k, v in label_map.items() if v == label][0]
        
        # Intra-class distances
        intra_dists = distances[mask][:, mask]
        intra_dists = intra_dists[np.triu_indices_from(intra_dists, k=1)]
        
        # Inter-class distances
        inter_dists = distances[mask][:, ~mask].flatten()
        
        results[phoneme] = {
            'intra_mean': float(intra_dists.mean()) if len(intra_dists) > 0 else 0,
            'intra_std': float(intra_dists.std()) if len(intra_dists) > 0 else 0,
            'inter_mean': float(inter_dists.mean()),
            'inter_std': float(inter_dists.std()),
            'separation_ratio': float(inter_dists.mean() / (intra_dists.mean() + 1e-8))
        }
    
    # Save results
    with open(output_dir / 'distance_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot distance distributions
    plt.figure(figsize=(10, 6))
    
    all_intra = []
    all_inter = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if labels[i] == labels[j]:
                all_intra.append(distances[i, j])
            else:
                all_inter.append(distances[i, j])
    
    plt.hist(all_intra, bins=50, alpha=0.5, label='Intra-class', density=True)
    plt.hist(all_inter, bins=50, alpha=0.5, label='Inter-class', density=True)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Distribution of Embedding Distances')
    plt.legend()
    plt.savefig(output_dir / 'distance_distributions.png', dpi=300)
    plt.close()
    
    return results


def find_confusable_pairs(embeddings, labels, label_map, output_dir, top_k=10):
    """Find most confusable phoneme pairs based on embedding distance."""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Compute mean embeddings per class
    class_embeddings = {}
    for label in unique_labels:
        mask = labels == label
        class_embeddings[label] = embeddings[mask].mean(axis=0)
    
    # Compute pairwise distances between class centers
    confusion_scores = []
    
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i >= j:  # Skip diagonal and lower triangle
                continue
                
            dist = np.linalg.norm(class_embeddings[label1] - class_embeddings[label2])
            phoneme1 = [k for k, v in label_map.items() if v == label1][0]
            phoneme2 = [k for k, v in label_map.items() if v == label2][0]
            
            confusion_scores.append({
                'phoneme1': phoneme1,
                'phoneme2': phoneme2,
                'distance': float(dist)
            })
    
    # Sort by distance (most confusable = smallest distance)
    confusion_scores.sort(key=lambda x: x['distance'])
    
    # Save top confusable pairs
    with open(output_dir / 'confusable_pairs.json', 'w') as f:
        json.dump(confusion_scores[:top_k], f, indent=2)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    pairs = [f"{cs['phoneme1']}-{cs['phoneme2']}" for cs in confusion_scores[:top_k]]
    distances = [cs['distance'] for cs in confusion_scores[:top_k]]
    
    plt.barh(pairs[::-1], distances[::-1])
    plt.xlabel('Embedding Distance')
    plt.title(f'Top {top_k} Most Confusable Phoneme Pairs')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusable_pairs.png', dpi=300)
    plt.close()
    
    return confusion_scores


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    
    # Setup
    device = get_best_device()
    output_dir = Path(cfg.checkpoint_path).parent.parent if 'checkpoint_path' in cfg else Path('outputs/evaluation')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger = create_logger(output_dir / "logs")
    logger.info("Starting evaluation...")
    
    # Load checkpoint
    checkpoint_path = cfg.get('checkpoint_path', 'outputs/latest/checkpoints/checkpoint_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    
    # Load model
    model_config = checkpoint['config']['model'] if 'config' in checkpoint else cfg.model
    model = model_registry.create(model_config['type'], model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    data_path = Path(cfg.data.data_path)
    file_paths, labels, label_map, metadata = parse_dataset(data_path, logger)
    
    # Create dataset (no augmentation for evaluation)
    feature_extractor = build_feature_extractor(cfg.data.feature_extractor)
    dataset = PhonemeContrastiveDataset(
        file_paths=file_paths,
        labels=labels,
        metadata=metadata,
        feature_extractor=feature_extractor,
        augmentation_pipeline=None,
        config=dict(cfg.data),
        mode='val'
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings, labels, metadata = extract_embeddings(model, dataloader, device)
    logger.info(f"Extracted {len(embeddings)} embeddings")
    
    # Generate visualizations
    logger.info("Generating confusion matrix...")
    cm, report = plot_confusion_matrix(embeddings, labels, label_map, output_dir)
    
    logger.info("Creating t-SNE visualization...")
    tsne_embeddings = plot_tsne(embeddings, labels, metadata, label_map, output_dir)
    
    logger.info("Analyzing distances...")
    distance_results = analyze_distances(embeddings, labels, label_map, output_dir)
    
    logger.info("Finding confusable pairs...")
    confusable_pairs = find_confusable_pairs(embeddings, labels, label_map, output_dir)
    
    logger.info(f"Evaluation complete! Results saved to: {output_dir}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Overall k-NN Accuracy: {report['accuracy']:.3f}")
    print(f"\nTop 5 Most Confusable Pairs:")
    for i, pair in enumerate(confusable_pairs[:5]):
        print(f"{i+1}. {pair['phoneme1']} <-> {pair['phoneme2']} (distance: {pair['distance']:.3f})")


if __name__ == "__main__":
    main()