import ioh
import numpy as np

def simple_random_search(problem, n_iterations =100, n_particles =20):
    best_x = None
    best_y = float('inf')

    for i in range(n_iterations):

        x =np.random.uniform(-5, 5, problem.meta_data.n_variables)
        y = problem(x)

        if y <best_y:
            best_y = y
            best_x = x
        
        return best_x, best_y
    
problem = ioh.get_problem(1, dimension=5, instance=1)
best_x, best_y = simple_random_search(problem)
print(f"Best solution found: {best_y:.4f}")
print(f"Optimum value: {problem.optimum.y:.4f}")