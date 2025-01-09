import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
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


def predict_next_frame_and_evaluate(frames_dir, flow_dir, output_dir):
    """
    Prend un répertoire de frames originales et un flux optique pour prédire les frames suivantes 
    et évaluer les erreurs par rapport aux frames réelles.

    Parameters:
        frames_dir (str): Répertoire contenant les frames originales.
        flow_dir (str): Répertoire contenant les flux optiques (.flo).
        output_dir (str): Répertoire pour sauvegarder les frames prédites.
    
    Returns:
        list of float: Liste des erreurs MSE pour chaque prédiction.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    flow_paths = sorted([os.path.join(flow_dir, f) for f in os.listdir(flow_dir)
                         if f.lower().endswith('.flo')])

    if len(frame_paths) - 1 != len(flow_paths):
        raise ValueError("Le nombre de fichiers .flo doit correspondre au nombre de transitions entre frames." + str(len(frame_paths)) + " " + str(len(flow_paths)))

    mse_list = []

    for i in range(len(flow_paths)):
        # Lire la frame actuelle et le flux optique
        current_frame = np.array(Image.open(frame_paths[i])).astype(np.float32) / 255.0
        optical_flow = read_flo_file(flow_paths[i])

        # Prévoir la prochaine frame en utilisant le flux optique
        height, width, _ = current_frame.shape
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        next_frame_coords_x = (grid_x + optical_flow[..., 0]).clip(0, width - 1).astype(np.float32)
        next_frame_coords_y = (grid_y + optical_flow[..., 1]).clip(0, height - 1).astype(np.float32)

        next_frame_predicted = cv2.remap(current_frame, next_frame_coords_x, next_frame_coords_y,
                                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Lire la vraie prochaine frame
        next_frame_actual = np.array(Image.open(frame_paths[i + 1])).astype(np.float32) / 255.0

        # Calculer la MSE entre la frame prédite et la vraie frame
        mse = np.mean((next_frame_predicted - next_frame_actual) ** 2)
        mse_list.append(mse)

        # Sauvegarder la frame prédite
        predicted_frame_path = os.path.join(output_dir, f"predicted_frame_{i + 1:04d}.png")
        Image.fromarray((next_frame_predicted * 255).astype(np.uint8)).save(predicted_frame_path)

        # print(f"Frame {i + 1}: MSE = {mse:.6f}, sauvegardée dans {predicted_frame_path}")

    return mse_list



def predict_next_frame_and_evaluate_videoflow(frames_dir, flow_dir, output_dir):
    """
    Prend un répertoire de frames originales et un flux optique (sous forme d'images .jpg) 
    pour prédire les frames suivantes et évaluer les erreurs par rapport aux frames réelles.

    Parameters:
        frames_dir (str): Répertoire contenant les frames originales.
        flow_dir (str): Répertoire contenant les flux optiques (images .jpg).
        output_dir (str): Répertoire pour sauvegarder les frames prédites.
    
    Returns:
        list of float: Liste des erreurs MSE pour chaque prédiction.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    flow_paths = sorted([os.path.join(flow_dir, f) for f in os.listdir(flow_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Gestion des tailles incohérentes
    if len(frame_paths) > len(flow_paths) + 1:
        frame_paths = frame_paths[:len(flow_paths) + 1]  # Ajuster les frames
    elif len(flow_paths) > len(frame_paths) - 1:
        flow_paths = flow_paths[:len(frame_paths) - 1]  # Ajuster les flux optiques

    # Vérification finale
    if len(frame_paths) - 1 != len(flow_paths):
        raise ValueError("Le nombre de flux optiques doit correspondre au nombre de transitions entre frames."
                         + f" Frames: {len(frame_paths)}, Flows: {len(flow_paths)}")

    mse_list = []

    for i in range(len(flow_paths)):
        # Lire la frame actuelle et le flux optique
        current_frame = np.array(Image.open(frame_paths[i])).astype(np.float32) / 255.0
        optical_flow = np.array(Image.open(flow_paths[i])).astype(np.float32) / 255.0

        # Séparer le flux optique en composantes (u, v)
        optical_flow_u = optical_flow[..., 0] * 2 - 1  # Normalisation entre -1 et 1
        optical_flow_v = optical_flow[..., 1] * 2 - 1  # Normalisation entre -1 et 1

        # Prévoir la prochaine frame en utilisant le flux optique
        height, width, _ = current_frame.shape
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        next_frame_coords_x = (grid_x + optical_flow_u).clip(0, width - 1).astype(np.float32)
        next_frame_coords_y = (grid_y + optical_flow_v).clip(0, height - 1).astype(np.float32)

        next_frame_predicted = cv2.remap(current_frame, next_frame_coords_x, next_frame_coords_y,
                                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Lire la vraie prochaine frame
        next_frame_actual = np.array(Image.open(frame_paths[i + 1])).astype(np.float32) / 255.0

        # Calculer la MSE entre la frame prédite et la vraie frame
        mse = np.mean((next_frame_predicted - next_frame_actual) ** 2)
        mse_list.append(mse)

        # Sauvegarder la frame prédite
        predicted_frame_path = os.path.join(output_dir, f"predicted_frame_{i + 1:04d}.png")
        Image.fromarray((next_frame_predicted * 255).astype(np.uint8)).save(predicted_frame_path)

        # print(f"Frame {i + 1}: MSE = {mse:.6f}, sauvegardée dans {predicted_frame_path}")

    return mse_list

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


    # this to evaluate with GITW DATASET
    # mse_errors_1 = predict_next_frame_and_evaluate("../GITW_selection_copy_reduced/Bowl/", "../GITW_selection_copy_reduced/Bowl_PCA/" , "../GITW_selection_copy_reduced/Bowl_PCA/pred/")
    # mse_errors_2 = predict_next_frame_and_evaluate("../GITW_selection_copy_reduced/Rice/", "../GITW_selection_copy_reduced/Rice_PCA/" , "../GITW_selection_copy_reduced/Rice_PCA/pred/")
    # mse_errors_3 = predict_next_frame_and_evaluate("../GITW_selection_copy_reduced/CanOFCocaCola/", "../GITW_selection_copy_reduced/CanOFCocaCola_PCA/" , "../GITW_selection_copy_reduced/CanOFCocaCola_PCA/pred/")
    # mse_errors_1 = predict_next_frame_and_evaluate_videoflow("../GITW_selection_copy_reduced/Bowl/", "../GITW_selection_copy_reduced/Bowl_vis_full" , "../GITW_selection_copy_reduced/Bowl_vis_full/pred/")
    # mse_errors_2 = predict_next_frame_and_evaluate_videoflow("../GITW_selection_copy_reduced/Rice/", "../GITW_selection_copy_reduced/Bowl_vis_full" , "../GITW_selection_copy_reduced/Rice_vis_full/pred/")
    # mse_errors_3 = predict_next_frame_and_evaluate_videoflow("../GITW_selection_copy_reduced/CanOFCocaCola/", "../GITW_selection_copy_reduced/CanOFCocaCola_vis_full" , "../GITW_selection_copy_reduced/CanOFCocaCola_vis_full/pred/")
    # je veux affichier les erreurs sous forme de histogramme pour les trois jeux de données

    # Moyenne des MSE pour chaque jeu de données
    # mse_means = [
    #     np.mean(mse_errors_1),
    #     np.mean(mse_errors_2),
    #     np.mean(mse_errors_3)
    # ]

    # # Labels des jeux de données
    # labels = ['GITW/Bowl', 'GITW/Rice', 'GITW/CanOFCocaCola']
    # x = np.arange(len(labels))  # Positions des barres

    # # Création de l'histogramme
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.bar(x, mse_means, width=0.5, color=['blue', 'orange', 'green'])

    # # Ajout des labels et titre
    # ax.set_xlabel('Jeux de données')
    # ax.set_ylabel('MSE moyenne')
    # ax.set_title('MSE des prédictions de frames par jeu de données')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)

    # # Affichage
    # plt.tight_layout()
    # plt.show()

    # Générer les étiquettes pour SINTEL DATASET
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
