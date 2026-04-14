import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── Settings ──────────────────────────────────────────────
INPUT_DIR = "data_subset"
OUTPUT_DIR = "dataset_from_subset"
K_CLUSTERS = 5
N_PARTICLES = 20   # try 20 first
# ──────────────────────────────────────────────────────────

def load_trajectory(path):
    """Load file and convert to (iterations, particles, dim)."""
    
    data = np.loadtxt(path, skiprows=1)
    
    # Last columns = position (x0, x1, ...)
    dim = data.shape[1] - 2
    positions = data[:, -dim:]
    
    total_points = len(positions)
    
    # Compute number of full iterations
    n_iterations = total_points // N_PARTICLES
    
    # Trim extra rows if needed
    positions = positions[:n_iterations * N_PARTICLES]
    
    trajectory = positions.reshape(n_iterations, N_PARTICLES, dim)
    
    return trajectory

def get_fingerprint(trajectory):
    """Convert trajectory to cluster matrix."""
    n_iterations = trajectory.shape[0]
    fingerprint = np.zeros((n_iterations, K_CLUSTERS))
    
    for t in range(n_iterations):
        particles = trajectory[t]
        
        kmeans = KMeans(n_clusters=K_CLUSTERS, n_init=10)
        labels = kmeans.fit_predict(particles)
        
        for k in range(K_CLUSTERS):
            fingerprint[t, k] = np.sum(labels == k) / len(labels)
    
    return fingerprint

def save_image(fingerprint, path):
    """Save fingerprint as grayscale image."""
    plt.figure(figsize=(2, 2))
    plt.imshow(fingerprint.T, cmap='gray', aspect='auto', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=64, bbox_inches='tight', pad_inches=0)
    plt.close()

# ── Main loop ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".txt") and not file.startswith("Base"):
        continue
    file_path = os.path.join(INPUT_DIR, file)
    
    # Skip non-files
    if not os.path.isfile(file_path):
        continue
    
    # Extract algorithm name
    algo = file.split('_')[0]
    
    algo_dir = os.path.join(OUTPUT_DIR, algo)
    os.makedirs(algo_dir, exist_ok=True)
    
    try:
        trajectory = load_trajectory(file_path)
        fingerprint = get_fingerprint(trajectory)
        
        save_path = os.path.join(algo_dir, file + ".png")
        save_image(fingerprint, save_path)
        
        print(f"Saved: {save_path}")
    
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("Done!")