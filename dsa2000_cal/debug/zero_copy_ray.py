import ray
import numpy as np

if __name__ == '__main__':

    ray.init(address='local')

    original_array = np.random.rand(10, 10)

    ray_ref = ray.put(original_array)

    retrieved_array = ray.get(ray_ref)

    # Verify zero-copy
    if np.shares_memory(original_array, retrieved_array):
        print("Zero-copy confirmed!")
    else:
        print("Data was copied!")