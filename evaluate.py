import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
def compute_metrics(flow1, flow2):
    """
    Compare two optical flows using aEPE, aAE, and MSE.
    
    Parameters:
        flow1: np.ndarray (H, W, 2) - First optical flow (u, v)
        flow2: np.ndarray (H, W, 2) - Second optical flow (u, v)
    
    Returns:
        aEPE: float - Average Endpoint Error
        aAE: float - Average Angular Error (radians)
        MSE: float - Mean Squared Error
    """
    # Ensure the flows have the same shape
    assert flow1.shape == flow2.shape, f"Flow dimensions must match ({flow1.shape} != {flow2.shape}) hh"
    
    # Compute the differences
    diff = flow1 - flow2
    
    # aEPE
    epe = np.sqrt(np.sum(diff**2, axis=-1))  # EPE per pixel
    aEPE = np.mean(epe)  # Average EPE
    
    # aAE
    norm1 = np.linalg.norm(flow1, axis=-1)  # Norm of flow1 per pixel
    norm2 = np.linalg.norm(flow2, axis=-1)  # Norm of flow2 per pixel
    dot_product = np.sum(flow1 * flow2, axis=-1)  # Dot product
    cos_angle = np.clip(dot_product / (norm1 * norm2 + 1e-8), -1, 1)  # Avoid division by zero
    ae = np.arccos(cos_angle)  # Angular error per pixel
    ae[norm1 == 0] = 0  # Set AE to 0 where norm1 is zero
    ae[norm2 == 0] = 0  # Set AE to 0 where norm2 is zero
    aAE = np.mean(ae)  # Average Angular Error
    
    # MSE
    mse = np.mean(np.sum(diff**2, axis=-1))  # Mean Squared Error
    
    return aEPE, aAE, mse



def read_flo_file(filepath):
    """
    Lit un fichier .flo et retourne un tableau numpy représentant le flux optique.
    Format: https://vision.middlebury.edu/flow/floweval-iccv07.pdf

    Parameters:
        filepath (str): Chemin du fichier .flo

    Returns:
        np.ndarray: Flux optique (H, W, 2)
    """
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'PIEH':
            raise ValueError(f"Le fichier {filepath} n'est pas un fichier .flo valide.")
        width = np.frombuffer(f.read(4), np.int32)[0]
        height = np.frombuffer(f.read(4), np.int32)[0]
        data = np.frombuffer(f.read(), np.float32).reshape((height, width, 2))
    return data


def compute_metrics_from_flo_files(flow1_dir, flow2_dir):
    """
    Calcule les métriques entre deux ensembles de fichiers .flo.

    Parameters:
        flow1_dir (str): Répertoire contenant les fichiers .flo pour le premier flux
        flow2_dir (str): Répertoire contenant les fichiers .flo pour le deuxième flux
    
    Returns:
        tuple: Moyenne des aEPE, aAE et MSE
    """
    flow1_files = sorted([
        os.path.join(flow1_dir, f) for f in os.listdir(flow1_dir)
        if os.path.isfile(os.path.join(flow1_dir, f)) and f.lower().endswith('.flo')
    ])
    flow2_files = sorted([
        os.path.join(flow2_dir, f) for f in os.listdir(flow2_dir)
        if os.path.isfile(os.path.join(flow2_dir, f)) and f.lower().endswith('.flo')
    ])

    # Vérifie que le nombre de fichiers correspond
    assert len(flow1_files) == len(flow2_files), f"Le nombre de fichiers doit correspondre ({len(flow1_files)} != {len(flow2_files)})"
    
    aepe, aae, mse = [], [], []
    for file1, file2 in zip(flow1_files, flow2_files):
        flow1 = read_flo_file(file1)
        flow2 = read_flo_file(file2)
        aEPE, aAE, MSE = compute_metrics(flow1, flow2)
        aepe.append(aEPE)
        aae.append(aAE)
        mse.append(MSE)
    
    return np.mean(aepe), np.mean(aae), np.mean(mse)

def compute_metrics_from_frames(flow1_dir, flow2_dir):
    frames1 = sorted([
        os.path.join(flow1_dir, f) for f in os.listdir(flow1_dir)
        if os.path.isfile(os.path.join(flow1_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames2 = sorted([
        os.path.join(flow2_dir, f) for f in os.listdir(flow2_dir)
        if os.path.isfile(os.path.join(flow2_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames2 = frames2[1:]  # Ignore the first frame
    frames2 = frames2[1: min(len(frames1) + 1, len(frames2) + 1)]  # Ignore the first frame
    if len(frames1) < len(frames2):
        frames2 = frames2[1:len(frames1)]
    elif len(frames1) > len(frames2):
        frames1 = frames1[:len(frames2)]

    assert len(frames1) == len(frames2), f"Le nombre de frames doit correspondre ({len(frames1)} != {len(frames2)})"
    aepe, aae, mse = [], [], []
    for i in range(len(frames1)):
        flow1 = np.array(Image.open(frames1[i])).astype(np.float32) / 255.0
        flow2 = np.array(Image.open(frames2[i])).astype(np.float32) / 255.0
        aEPE, aAE, MSE = compute_metrics(flow1, flow2)
        aepe.append(aEPE)
        aae.append(aAE)
        mse.append(MSE)
    aepe = np.mean(aepe)
    aae = np.mean(aae)
    mse = np.mean(mse)
    return aepe, aae, mse


def process_datasets(dataset_dirs, method):
    """
    Calcule les métriques pour trois jeux de données en utilisant une méthode donnée.

    Parameters:
        dataset_dirs (list of tuples): Liste des paires de répertoires (flow1_dir, flow2_dir).
        method (str): Méthode à utiliser ('dl' ou 'classique').

    Returns:
        list of dict: Liste contenant les résultats des métriques pour chaque jeu de données.
    """
    results = []
    for i, (flow1_dir, flow2_dir) in enumerate(dataset_dirs):
        print(f"Processing dataset {i+1} using {method} method...")
        if method == 'dl':
            aepe, aae, mse = compute_metrics_from_frames(flow1_dir, flow2_dir)
        elif method == 'classique':
            aepe, aae, mse = compute_metrics_from_flo_files(flow1_dir, flow2_dir)
        else:
            raise ValueError("Méthode non reconnue. Utilisez 'dl' ou 'classique'.")
        results.append({'aEPE': aepe, 'aAE': aae, 'MSE': mse})
    return results



def plot_metrics(metrics, labels):
    """
    Génère un histogramme des métriques pour plusieurs jeux de données.

    Parameters:
        metrics (list of dict): Liste des métriques (aEPE, aAE, MSE) pour chaque jeu de données.
        labels (list of str): Liste des étiquettes pour chaque jeu de données.
    """
    aepe_values = [m['aEPE'] for m in metrics]
    aae_values = [m['aAE'] for m in metrics]
    mse_values = [m['MSE'] for m in metrics]

    x = np.arange(len(labels))  # Indices des groupes
    width = 0.25  # Largeur des barres

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, aepe_values, width, label='aEPE')
    ax.bar(x, aae_values, width, label='aAE')
    ax.bar(x + width, mse_values, width, label='MSE')

    # Configuration de l'axe
    ax.set_xlabel('Jeux de données')
    ax.set_ylabel('Valeurs des métriques')
    ax.set_title('Métriques des flux optiques par jeu de données')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 8:
        print("Usage: python script.py <flow1_dir1> <flow2_dir1> <flow1_dir2> <flow2_dir2> <flow1_dir3> <flow2_dir3> <method>")
        sys.exit(1)

    # Récupération des répertoires pour les trois jeux de données
    flow1_dir1, flow2_dir1 = sys.argv[1], sys.argv[2]
    flow1_dir2, flow2_dir2 = sys.argv[3], sys.argv[4]
    flow1_dir3, flow2_dir3 = sys.argv[5], sys.argv[6]
    method = sys.argv[7]  # 'dl' ou 'classique'

    # Traiter les trois jeux de données
    dataset_dirs = [(flow1_dir1, flow2_dir1), (flow1_dir2, flow2_dir2), (flow1_dir3, flow2_dir3)]
    metrics = process_datasets(dataset_dirs, method)

    # Générer les étiquettes pour les jeux de données
    labels = ['GITW/Bowl', 'GITW/Rice', 'GITW/CanOFCocaCola']

    aepe_values = [m['aEPE'] for m in metrics]
    aae_values = [m['aAE'] for m in metrics]
    # mse_values = [m['MSE'] for m in metrics]

    x = np.arange(len(labels))  # Indices des groupes
    width = 0.25  # Largeur des barres

    fig, ax = plt.subplots(figsize=(7, 4))
    # for i in range (len(labels)):
    #     ax.bar(x - width, aepe_values, width, label='aEPE')
    #     ax.bar(x, aae_values, width, label='aAE')
    # ax.bar(x , mse_values, width, label='MSE')
    ax.bar(x - width, aepe_values, width, label='aEPE')
    ax.bar(x, aae_values, width, label='aAE')
    # Configuration de l'axe
    ax.set_xlabel('Jeux de données')
    ax.set_ylabel('Valeurs des métriques')
    ax.set_title('Métriques des flux optiques par jeu de données avec la méthode ' + method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()
