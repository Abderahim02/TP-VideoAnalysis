import os
import cv2
import numpy as np

def get_frame_paths(directory):
    """
    Récupère les chemins des frames dans un répertoire trié par ordre alphabétique.
    """
    frame_paths = [os.path.join(directory, fname) for fname in sorted(os.listdir(directory))]
    return [path for path in frame_paths if os.path.isfile(path)]

def compute_optical_flow_pca(flow_dir, output_dir):
    """
    Génère le flux optique à partir d'un répertoire contenant des frames avec PCAFlow.
    
    Parameters:
        flow_dir (str): Chemin vers le répertoire contenant les frames.
        output_dir (str): Chemin vers le répertoire où enregistrer les flux optiques.
    """
    # Vérifie si les dossiers existent
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Récupère les chemins des frames
    frame_paths = get_frame_paths(flow_dir)
    if len(frame_paths) < 2:
        raise ValueError("Le répertoire doit contenir au moins deux frames pour calculer le flux optique.")
    
    # Initialise PCAFlow
    pca_flow = cv2.optflow.createOptFlow_PCAFlow()
    
    # Itère sur les paires consécutives de frames
    for i in range(len(frame_paths) - 1):
        frame1_path = frame_paths[i]
        frame2_path = frame_paths[i + 1]
        
        # Charge les frames en niveaux de gris
        frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        
        if frame1 is None or frame2 is None:
            print(f"Erreur lors du chargement des frames : {frame1_path}, {frame2_path}")
            continue
        
        # Calcule le flux optique
        flow = pca_flow.calc(frame1, frame2, None)
        
        # Enregistre le flux optique
        flow_filename = f"flow_{i + 1:04d}_to_{i+2:04d}.flo"
        flow_path = os.path.join(output_dir, flow_filename)
        write_flow(flow, flow_path)
        # print(f"Flux optique enregistré : {flow_path}")

def write_flow(flow, filename):
    """
    Enregistre un flux optique au format .flo.
    """
    with open(filename, 'wb') as f:
        # Header .flo
        f.write(b'PIEH')
        # Dimensions
        f.write(np.array(flow.shape[1], dtype=np.int32).tobytes())  # Width
        f.write(np.array(flow.shape[0], dtype=np.int32).tobytes())  # Height
        # Données
        f.write(flow.astype(np.float32).tobytes())

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    frame_directory = sys.argv[1]  # Chemin du répertoire des frames
    output_directory = sys.argv[2]  # Chemin du répertoire de sortie
    
    compute_optical_flow_pca(frame_directory, output_directory)
