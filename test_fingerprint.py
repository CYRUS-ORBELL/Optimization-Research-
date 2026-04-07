import numpy as np
import pyswarms as ps
import ioh
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

problem = ioh.get_problem(1, dimension=5, instance=1)
trajectory = []

def objective(particles):
    trajectory.append(particles.copy())
    return np.array([problem(p) for p in particles])

problem.reset()
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(
    n_particles=20,
    dimensions=5,
    options=options,
    bounds=(np.full(5, -5.0), np.full(5, 5.0))
)
optimizer.optimize(objective, iters=100, verbose=False)
trajectory = np.array(trajectory)  # (100, 20, 5)

# For each iteration, cluster the 20 particles into K clusters
# and record what fraction of particles are in each cluster
K = 5  # number of clusters
n_iterations = trajectory.shape[0]
fingerprint = np.zeros((n_iterations, K))

for t in range(n_iterations):
    particles = trajectory[t]  # shape (20, 5)
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(particles)
    for k in range(K):
        fingerprint[t, k] = np.sum(labels == k) / len(labels)

print(f"Fingerprint matrix shape: {fingerprint.shape}")  # should be (100, 5)

# Render as grayscale image
plt.figure(figsize=(6, 4))
plt.imshow(fingerprint.T, aspect='auto', cmap='gray')
plt.xlabel('Iteration')
plt.ylabel('Cluster')
plt.title('PSO Fingerprint - Sphere Function')
plt.colorbar(label='Fraction of particles')
plt.savefig('pso_fingerprint.png', dpi=100)
plt.close()
print("Saved pso_fingerprint.png")