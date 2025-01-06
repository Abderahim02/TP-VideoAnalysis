import os
import subprocess

import shutil

# # Chemin du répertoire contenant les images
# input_dir = "../MPI-Sintel_selection/training/final/alley_2"
# output_dir = "../MPI-Sintel_selection/training/final/alley_2_vis/"

def chunked_inference(input_dir, output_dir):
    # Liste des fichiers dans le répertoire d'entrée
    image_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

    # Taille du chunk
    chunk_size = 10

    for i in range(0, len(image_files), chunk_size):
        # Sélectionner un chunk
        chunk = image_files[i:i + chunk_size]
        
        # Créer un sous-répertoire temporaire pour ce chunk
        chunk_dir = os.path.join(input_dir, f"chunk_{i//chunk_size}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Copier les fichiers dans le répertoire temporaire
        for file_name in chunk:
            src = os.path.join(input_dir, file_name)
            dst = os.path.join(chunk_dir, file_name)
            shutil.copy(src, dst)  # Copie réelle du fichier
        
        # Exécuter la commande pour ce chunk
        command = [
            "python", "-u", "inference.py",
            "--mode", "MOF",
            "--seq_dir", chunk_dir,
            "--vis_dir", chunk_dir + "vis"
        ]
        print(f"Processing chunk {i//chunk_size + 1}...")
        subprocess.run(command)
        
        # Supprimer le répertoire temporaire après le traitement
        # for file_name in os.listdir(chunk_dir):
        #     os.remove(os.path.join(chunk_dir, file_name))
        # os.rmdir(chunk_dir)

    print("Traitement terminé !")


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    input_dir = argv[0]
    output_dir = argv[1]
    chunked_inference(input_dir, output_dir)