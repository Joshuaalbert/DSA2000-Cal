import sympy as sp


def main():
    # Define symbols
    x, y, z = sp.symbols('x y z', real=True)
    omega, dt = sp.symbols('omega dt', real=True)
    alpha_pole, delta_pole = sp.symbols('alpha_pole delta_pole', real=True)

    # Define the rotation angle theta = omega*dt
    theta = omega * dt

    # Precompute trigonometric functions for the rotation angle
    c = sp.cos(theta)
    s = sp.sin(theta)
    v = 1 - c  # 1 - cos(theta)

    # Define the rotation axis in Cartesian coordinates
    n_x = sp.cos(delta_pole) * sp.cos(alpha_pole)
    n_y = sp.cos(delta_pole) * sp.sin(alpha_pole)
    n_z = sp.sin(delta_pole)

    # Construct the Rodrigues rotation matrix
    R = sp.Matrix([
        [c + n_x ** 2 * v, n_x * n_y * v - n_z * s, n_x * n_z * v + n_y * s],
        [n_y * n_x * v + n_z * s, c + n_y ** 2 * v, n_y * n_z * v - n_x * s],
        [n_z * n_x * v - n_y * s, n_z * n_y * v + n_x * s, c + n_z ** 2 * v]
    ])

    # Define the original point in Cartesian coordinates
    p = sp.Matrix([x, y, z])

    # Compute the rotated point
    p_rot = R * p

    # Simplify the result
    p_rot_simplified = sp.simplify(p_rot)

    # Pretty-print the result
    sp.pprint(p_rot_simplified)

    # CSE
    cse = sp.cse(p_rot_simplified)
    print(cse)
    for i in range(len(cse[0])):
        print(f"{cse[0][i][0]} = {cse[0][i][1]}")

    for sym, expression in zip(['x_out', 'y_out', 'z_out'], cse[1][0]):
        print(f"{sym} = {expression}")


if __name__ == '__main__':
    main()
