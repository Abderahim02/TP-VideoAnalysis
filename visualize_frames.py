import os
from PIL import Image
import matplotlib
matplotlib.use("webagg")
import matplotlib.pyplot as plt

class FrameNavigator:
    def __init__(self, direct_frames, reverse_frames, real_frames):
        """Initialise le visualiseur avec trois listes de frames (directes, inversées, et réelles)."""
        self.direct_frames = direct_frames
        self.reverse_frames = reverse_frames
        self.real_frames = real_frames

        self.current_idx_direct = 0 
        self.current_idx_reverse = 0 
        self.current_idx_real = 0  

        # Créer la figure et afficher les trois frames
        self.fig, (self.ax_direct, self.ax_reverse, self.ax_real) = plt.subplots(1, 3, figsize=(12, 8))
        self.ax_direct.axis('off') 
        self.ax_reverse.axis('off')  
        self.ax_real.axis('off')  

        # Afficher la première image dans les trois axes
        self.img_direct = self.ax_direct.imshow(Image.open(self.direct_frames[self.current_idx_direct]))
        self.ax_direct.set_title(f"Direct: Frame {self.current_idx_direct + 1}/{len(self.direct_frames)}", fontsize=16)

        self.img_real = self.ax_real.imshow(Image.open(self.real_frames[self.current_idx_real]))
        self.ax_real.set_title(f"Real: Frame {self.current_idx_real + 1}/{len(self.real_frames)}", fontsize=16)

        # Connecter les événements de clic à la fonction de gestion
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def update_frame(self):
        """Met à jour les images affichées pour les trois sens."""
        # Mise à jour des images directes
        self.img_direct.set_array(Image.open(self.direct_frames[self.current_idx_direct]))
        self.ax_direct.set_title(f"Direct: Frame {self.current_idx_direct + 1}/{len(self.real_frames)}", fontsize=16)

        self.img_real.set_array(Image.open(self.real_frames[self.current_idx_real]))
        self.ax_real.set_title(f"Real: Frame {self.current_idx_real + 1}/{len(self.real_frames)}", fontsize=16)
        self.fig.canvas.draw()

    def on_click(self, event):
        """Gère les clics de souris pour naviguer entre les frames."""
        # Si on clique dans la partie gauche (Direct)
        if event.inaxes == self.ax_direct:
            if event.button == 1:  # Clic gauche : avancer
                self.current_idx_direct = (self.current_idx_direct + 1) % len(self.direct_frames)
            elif event.button == 3:  # Clic droit : reculer
                self.current_idx_direct = (self.current_idx_direct - 1) % len(self.direct_frames)

        # Si on clique dans la partie droite (Real)
        elif event.inaxes == self.ax_real:
            if event.button == 1:  # Clic gauche : avancer
                self.current_idx_real = (self.current_idx_real + 1) % len(self.real_frames)
            elif event.button == 3:  # Clic droit : reculer
                self.current_idx_real = (self.current_idx_real - 1) % len(self.real_frames)

        # Mettre à jour les images après le clic
        self.update_frame()

    def show(self):
        """Affiche le visualiseur."""
        self.update_frame()
        plt.show()

def load_frames(seq_dir, real_dir):
    """Charge les frames depuis deux répertoires donnés (pour les frames directes/inversées et réelles)."""
    frames = sorted([
        os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
        if os.path.isfile(os.path.join(seq_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    real_frames = sorted([
        os.path.join(real_dir, f) for f in os.listdir(real_dir)
        if os.path.isfile(os.path.join(real_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # Séparer les frames en deux listes : une pour le sens direct et l'autre pour l'inverse
    direct_frames = []
    reverse_frames = []
    direct_frames, reverse_frames = frames, frames
    return direct_frames, reverse_frames, real_frames

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    seq_dir = sys.argv[1]  # Chemin du répertoire des frames 
    real_dir = sys.argv[2]  # Chemin du répertoire des frames réelles

    direct_frames, reversed_frames, real_frames = load_frames(seq_dir, real_dir)

    if not direct_frames or not real_frames:
        print("Aucune image trouvée dans le répertoire spécifié.")
    else:
        print(f"{len(direct_frames) + len(reversed_frames)} frames directes et {len(real_frames)} frames réelles chargées.")
        navigator = FrameNavigator(direct_frames, reversed_frames, real_frames)
        navigator.show()
