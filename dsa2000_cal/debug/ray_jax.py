import ray
import jax
import jax.numpy as jnp

if __name__ == '__main__':
    ray.init(address='local')

    a = jnp.ones((5,))
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.key(0)

    a_ref = ray.put(a)
    key1_ref = ray.put(key1)
    key2_ref = ray.put(key2)

    a_rec = ray.get(a_ref)
    key1_rec = ray.get(key1_ref)
    key2_rec = ray.get(key2_ref)


