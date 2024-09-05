import sympy as sp

def main():
    # Define variables for coefficients and coordinates
    a, b, c, d, e, f = sp.symbols('a b c d e f')
    x1, y1, z1 = sp.symbols('x1 y1 z1')
    x2, y2, z2 = sp.symbols('x2 y2 z2')
    x3, y3, z3 = sp.symbols('x3 y3 z3')
    x4, y4, z4 = sp.symbols('x4 y4 z4')
    x5, y5, z5 = sp.symbols('x5 y5 z5')
    x6, y6, z6 = sp.symbols('x6 y6 z6')

    # Quadratic form design matrix

    A = sp.Matrix([
        [x1**2, y1**2, x1*y1, x1, y1, 1],
        [x2**2, y2**2, x2*y2, x2, y2, 1],
        [x3**2, y3**2, x3*y3, x3, y3, 1],
        [x4**2, y4**2, x4*y4, x4, y4, 1],
        [x5**2, y5**2, x5*y5, x5, y5, 1],
        [x6**2, y6**2, x6*y6, x6, y6, 1]
        ])
    # Invert
    A_inv = A.inv()
    print(A_inv)

    # Solve
    b = sp.Matrix([z1, z2, z3, z4, z5, z6])
    solution = A_inv * b

    # Print the solution
    print(solution)

    # Evaluate the quadratic form with the solution a point x,y
    x,y = sp.symbols('x y')
    quadratic_form_solution = solution[0]*x**2 + solution[1]*y**2 + solution[2]*x*y + solution[3]*x + solution[4]*y + solution[5]
    print(quadratic_form_solution)

    # Simplify that
    quadratic_form_solution_simplified = sp.simplify(quadratic_form_solution)
    print(quadratic_form_solution_simplified)

if __name__ == '__main__':
    main()
