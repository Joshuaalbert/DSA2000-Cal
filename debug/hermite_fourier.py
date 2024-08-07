def main_hermite():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import hermite

    # Define the mesh grid
    x = np.linspace(0, 1, 400)
    y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y)

    # Define the order of Hermite polynomials
    orders = [0, 1, 2, 3]

    # Create a figure
    plt.figure(figsize=(10, 10))

    # Plot Hermite basis functions
    for i, n in enumerate(orders):
        H_n = hermite(n)
        Z = H_n(X) * H_n(Y)
        plt.subplot(2, 2, i + 1)
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.title(f'Hermite Polynomial H_{n}(x)H_{n}(y)')
        plt.colorbar()

    plt.tight_layout()
    plt.show()


def main_lagurre():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import genlaguerre

    # Define the mesh grid
    x = np.linspace(0, 1, 400)
    y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y)

    # Define the order of Laguerre polynomials
    orders = [0, 1, 2, 3]

    # Create a figure
    plt.figure(figsize=(10, 10))

    # Plot Laguerre basis functions
    for i, n in enumerate(orders):
        L_n = genlaguerre(n, 0)
        Z = L_n(X) * L_n(Y)
        plt.subplot(2, 2, i + 1)
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.title(f'Laguerre Polynomial L_{n}(x)L_{n}(y)')
        plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main_lagurre()
