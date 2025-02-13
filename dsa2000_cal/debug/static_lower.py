import jax
import jax.numpy as jnp


def f(x, y):
    if y:
        return x
    return -x

def main():
    x = jnp.ones(1)
    y = True
    f_jit = jax.jit(f, static_argnames=['y']).lower(x,y).compile()
    print(f_jit(x,y))

if __name__ == '__main__':
    main()