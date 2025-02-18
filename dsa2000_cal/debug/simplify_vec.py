import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import xla

# Define the primitives
vec_p = core.Primitive('vec')
unvec_p = core.Primitive('unvec')

# Implementation of the primitives
def vec_impl(x):
    return x.reshape(-1)  # Example implementation

def unvec_impl(x, shape):
    return x.reshape(shape)  # Example implementation

vec_p.def_impl(vec_impl)
unvec_p.def_impl(unvec_impl)

# Abstract evaluation rules
def vec_abstract_eval(x):
    return core.ShapedArray((x.size,), x.dtype)

def unvec_abstract_eval(x, shape):
    return core.ShapedArray(shape, x.dtype)

vec_p.def_abstract_eval(vec_abstract_eval)
unvec_p.def_abstract_eval(unvec_abstract_eval)

# Define the actual operations
def vec(x):
    return vec_p.bind(x)

def unvec(x, shape):
    return unvec_p.bind(x, shape)

# Custom optimization rules
def vec_unvec_rule(primals, params):
    x, = primals
    return x

def unvec_vec_rule(primals, params):
    x, = primals
    return x

# Register optimization rules
jax.interpreters.ad.primitive_transposes[vec_p] = lambda cotangent, **params: cotangent
jax.interpreters.ad.primitive_transposes[unvec_p] = lambda cotangent, **params: cotangent

def vec_unvec_translation(c, x, **params):
    return x

def unvec_vec_translation(c, x, **params):
    return x

jax.interpreters.xla.translations[vec_p] = vec_unvec_translation
jax.interpreters.xla.translations[unvec_p] = unvec_vec_translation

# Example usage
x = jnp.array([[1, 2], [3, 4]])
vec_x = vec(x)
unvec_x = unvec(vec_x, x.shape)

print("Original x:")
print(x)
print("Vectorized x:")
print(vec_x)
print("Unvectorized x:")
print(unvec_x)
print("Simplified vec(unvec(x)):")
print(jax.core.eval_jaxpr(jax.make_jaxpr(vec)(unvec_x), [x]))
print("Simplified unvec(vec(x)):")
print(jax.core.eval_jaxpr(jax.make_jaxpr(unvec)(vec_x, x.shape), [x]))
