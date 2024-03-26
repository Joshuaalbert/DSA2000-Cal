import jax
import jax.numpy as jnp

def example_function(x):
    return jnp.sin(x) + jnp.cos(x)

if __name__ == '__main__':
    # Create a placeholder for input to generate HLO
    x = jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float32)
    xla_computation = jax.xla_computation(example_function)(x)
    hlo_text = xla_computation.as_hlo_text()

    with open("example.hlo", "w") as file:
        file.write(hlo_text)
