
import jax.numpy as jnp
from jax import Array

float_type = jnp.result_type(float)
int_type = jnp.result_type(int)
complex_type = jnp.result_type(complex)

PRNGKey = Array
FloatArray = Array
IntArray = Array
BoolArray = Array

def a_(x):
    return jnp.asarray(x)