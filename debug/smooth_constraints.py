import jax
import jax.numpy as jnp


def sigmoid(x):
    return jax.nn.sigmoid(x)


def f1(x):
    return jnp.square(x) / (1 + jnp.square(x))


def f2(x):
    return 0.5 * (1 + x / jnp.sqrt(1 + jnp.square(x)))


def f3(x):
    return (0.5 * jnp.sign(x)) * f1(x) + 0.5


def f4(x):
    return 0.5 * (x / (1 + jnp.sign(x) * x) + 1)


def f4_inverse(y):
    return jnp.where(y >= 0.5,
                     (1 - 2 * y) / (2 * (y - 1)),
                     1 - 1 / (2 * y)
                     )


def main():
    x = jnp.linspace(-10, 10, 100000)
    y_sigmoid = sigmoid(x)
    y_1 = f1(x)
    y_2 = f2(x)
    y_3 = f3(x)
    y_4 = f4(x)

    import pylab as plt
    plt.plot(x, y_sigmoid, label='sigmoid')
    plt.plot(x, y_1, label='f1')
    plt.plot(x, y_2, label='f2')
    plt.plot(x, y_3, label='f3')
    plt.plot(x, y_4, label='f4')
    plt.legend()
    plt.show()

    # cost analysis
    print(jax.jit(jax.vmap(sigmoid)).lower(x).compile().cost_analysis())
    print(jax.jit(jax.vmap(f1)).lower(x).compile().cost_analysis())
    print(jax.jit(jax.vmap(f2)).lower(x).compile().cost_analysis())
    print(jax.jit(jax.vmap(f3)).lower(x).compile().cost_analysis())
    print(jax.jit(jax.vmap(f4)).lower(x).compile().cost_analysis())

    # speed test
    import time
    for f in [sigmoid, f1, f2, f3, f4]:
        g = jax.jit(f).lower(x).compile()
        t0 = time.time()
        for _ in range(1000):
            g(x).block_until_ready()
        print(f.__name__, time.time() - t0)

    # broadcast
    # sigmoid 0.07168745994567871
    # f1 0.024517536163330078
    # f2 0.04795193672180176
    # f3 0.035570621490478516

    for f in [sigmoid, f1, f2, f3, f4]:
        g = jax.jit(jax.vmap(f)).lower(x).compile()
        t0 = time.time()
        for _ in range(1000):
            g(x).block_until_ready()
        print(f.__name__, time.time() - t0)

    # vmap
    # sigmoid 0.07032585144042969
    # f1 0.023139238357543945
    # f2 0.04231381416320801
    # f3 0.03059983253479004

    for f in [sigmoid, f1, f2, f3, f4]:
        g = jax.grad(f)
        print(g(0.))

    y = jnp.linspace(0.01, 0.99, 1000)
    plt.plot(y, f4_inverse(y))
    plt.show()
    print(f4_inverse(y))


if __name__ == '__main__':
    main()
