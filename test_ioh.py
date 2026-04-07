import ioh
import numpy as np

for func_id in range(1, 6):
    problem = ioh.get_problem(func_id, dimension=5, instance=1)
    x = np.random.uniform(-5, 5, 5)
    y = problem(x)
    print(f"f{func_id} - {problem.meta_data.name}: f(x) = {y:.4f}")