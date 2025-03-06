#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <cmath>

//------------------------------------------------------------------------------
// Helper Function: Compute Inverse Square Root of an SPD Matrix
//------------------------------------------------------------------------------
// Given a symmetric positive definite (SPD) covariance matrix Sigma,
// we compute Sigma^{-1/2} using its eigendecomposition.
Eigen::MatrixXd computeSigmaInvSqrt(const Eigen::MatrixXd &Sigma) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
    // Get eigenvalues and eigenvectors
    Eigen::VectorXd evals = es.eigenvalues();
    Eigen::MatrixXd evecs = es.eigenvectors();
    // Compute the diagonal matrix with entries 1/sqrt(lambda)
    Eigen::VectorXd invSqrtEvals = evals.array().sqrt().inverse();
    return evecs * invSqrtEvals.asDiagonal() * evecs.transpose();
}

//------------------------------------------------------------------------------
// Flattened Multi-step Levenberg-Marquardt Algorithm with Adaptive Trust Region
//------------------------------------------------------------------------------
// This function implements the algorithm described in the pseudocode.
// Inputs:
//   V         : Data vector.
//   F         : Model function F(x), returning a vector.
//   dF        : Function to compute the Jacobian dF/dx (at x).
//   Sigma     : Covariance matrix (assumed SPD).
//   x0        : Initial estimate for x.
//   M         : Frequency to recompute (cache) the Jacobian.
//   maxIter   : Maximum number of outer (LM) iterations.
//   maxIterCG : Maximum number of inner Conjugate Gradient (CG) iterations.
//   mu0       : Initial damping parameter.
//   gtol      : Termination tolerance on the gradient norm.
//   tol_CG    : Relative tolerance for the CG residual.
//   atol_CG   : Absolute tolerance for the CG residual.
//   M_precond : Preconditioner for CG (default is identity).
//   p_lower, p_upper, p_accept : Trust region parameters controlling step updates.
//
// Returns a pair containing the final estimate and the number of iterations.
std::pair<Eigen::VectorXd, int> flattenedLM(
    const Eigen::VectorXd &V,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &F,
    const std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> &dF,
    const Eigen::MatrixXd &Sigma,
    const Eigen::VectorXd &x0,
    int M,
    int maxIter,
    int maxIterCG,
    double mu0 = 1.0,
    double gtol = 1e-5,
    double tol_CG = 1e-5,
    double atol_CG = 0.0,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &M_precond =
         [](const Eigen::VectorXd &v) { return v; },
    double p_lower = 0.25,
    double p_upper = 0.75,
    double p_accept = 0.75
) {
    // -------------------------------------------------------------------------
    // Precompute Sigma^{-1/2} for forming the residual.
    // -------------------------------------------------------------------------
    Eigen::MatrixXd SigmaInvSqrt = computeSigmaInvSqrt(Sigma);

    // -------------------------------------------------------------------------
    // Define helper lambdas for computing the residual and the Jacobian.
    //
    // The residual is defined as:
    //   R(x) = Sigma^{-1/2} * (V - F(x))
    //
    // The Jacobian of R is:
    //   J = ∇_x R(x) = -Sigma^{-1/2} * (dF/dx)
    // -------------------------------------------------------------------------
    auto computeResidual = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        return SigmaInvSqrt * (V - F(x));
    };

    auto computeJacobian = [&](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
        return -SigmaInvSqrt * dF(x);
    };

    // -------------------------------------------------------------------------
    // INITIALISATION STEP
    // -------------------------------------------------------------------------
    // Set the initial estimate.
    Eigen::VectorXd x = x0;

    // Compute the residual R(x) and its squared norm Q.
    Eigen::VectorXd R = computeResidual(x);
    double Q = R.squaredNorm();

    // Compute and cache the Jacobian at x.
    Eigen::MatrixXd J = computeJacobian(x);

    // Compute the negative gradient: g = -J^T R.
    Eigen::VectorXd grad = -J.transpose() * R;
    double gnorm = grad.norm();

    // -------------------------------------------------------------------------
    // LINE SEARCH TO INITIALIZE DAMPING PARAMETER mu
    // -------------------------------------------------------------------------
    double mu = mu0;
    // Avoid division by zero: only if gnorm > 0.
    if (gnorm > 0) {
        while (true) {
            // Trial step along the gradient direction (scaled by mu/gnorm).
            Eigen::VectorXd x_trial = x + (mu / gnorm) * grad;
            double Q_trial = computeResidual(x_trial).squaredNorm();
            // If the trial step reduces the cost, break out.
            if (Q_trial <= Q) {
                break;
            }
            // Otherwise, reduce mu.
            mu /= 2.0;
        }
    }

    // -------------------------------------------------------------------------
    // Set up history for multi-step (linear forecast) CG guess.
    // delta_x_prev represents δx⁻¹ and delta_x_prev2 represents δx⁻².
    // -------------------------------------------------------------------------
    Eigen::VectorXd delta_x_prev = Eigen::VectorXd::Zero(x.size());
    Eigen::VectorXd delta_x_prev2 = Eigen::VectorXd::Zero(x.size());

    int k = 0;  // Outer iteration counter.
    bool done = (gnorm <= gtol || k >= maxIter);

    // -------------------------------------------------------------------------
    // MAIN ITERATION LOOP
    // -------------------------------------------------------------------------
    while (!done) {
        // --- Linear Forecast: Predict an initial guess for δx ---
        // Use the previous two steps: δx = 2 * δx⁻¹ - δx⁻².
        Eigen::VectorXd delta_x = 2.0 * delta_x_prev - delta_x_prev2;

        // --- Construct the A operator for the CG solver ---
        // We need to solve for δx in the linear system:
        //      A δx = grad, where A = JᵀJ + (gnorm/μ) I.
        // Instead of forming A explicitly, we define an operator.
        auto A_operator = [&](const Eigen::VectorXd &v) -> Eigen::VectorXd {
            return J.transpose() * (J * v) + (gnorm / mu) * v;
        };

        // --- Conjugate Gradient (CG) Solver ---
        // Using the current guess delta_x, we solve A * delta_x = grad.
        Eigen::VectorXd Ax = A_operator(delta_x);
        Eigen::VectorXd r = grad - Ax;          // Residual of the linear system.
        Eigen::VectorXd z = M_precond(r);         // Preconditioned residual.
        Eigen::VectorXd p = z;                    // Initial search direction.
        double gamma = r.dot(z);
        int k_CG = 0;

        // Define CG stopping criterion based on relative and absolute tolerances.
        double cgTolSquared = std::max(tol_CG * tol_CG * gnorm * gnorm, atol_CG * atol_CG);

        while (r.squaredNorm() > cgTolSquared && k_CG < maxIterCG) {
            // Cache the matrix-vector product q = A_operator(p)
            Eigen::VectorXd q = A_operator(p);
            double alpha = gamma / (p.dot(q) + 1e-12); // Add small term to avoid division by zero.

            // Update δx
            delta_x += alpha * p;
            // Update the residual
            r -= alpha * q;
            // Precondition the new residual
            z = M_precond(r);
            double gamma_new = r.dot(z);
            double beta = gamma_new / (gamma + 1e-12); // Avoid division by zero.
            // Update search direction
            p = z + beta * p;
            gamma = gamma_new;
            k_CG++;
        }
        // --- End of CG Solver ---

        // --- Trust Region Update ---
        // Compute the trial solution x' = x + δx and its cost Q' = ||R(x')||².
        Eigen::VectorXd x_trial = x + delta_x;
        double Q_trial = computeResidual(x_trial).squaredNorm();

        // Compute the convex (linearized) predicted decrease:
        //   δ_convex = Q - ||R(x) + Jδx||²
        double convexDecrease = Q - (computeResidual(x) + J * delta_x).squaredNorm();
        // Compute the actual decrease:
        double actualDecrease = Q - Q_trial;

        // Compute the trust-region ratio ρ.
        double rho = (convexDecrease > 0) ? (actualDecrease / convexDecrease) : -1.0;

        // Adjust the damping parameter μ based on the quality of the step.
        if (convexDecrease > 0 && (rho > p_lower && rho < p_upper)) {
            // When the predicted decrease is in an acceptable middle range,
            // we are confident enough to try larger steps.
            mu *= 2.0;
        } else {
            // Otherwise, decrease μ to be more conservative.
            mu /= 2.0;
        }

        // Accept the trial step if the model predicts a positive decrease and ρ exceeds p_accept.
        if (convexDecrease > 0 && rho > p_accept) {
            x = x_trial;
            Q = Q_trial;
        }

        // --- Jacobian Recalculation ---
        // Every M iterations, recompute and cache the Jacobian at the current x.
        if (k % M == 0) {
            J = computeJacobian(x);
        }

        // Recompute the residual and update the gradient.
        R = computeResidual(x);
        grad = -J.transpose() * R;
        gnorm = grad.norm();

        // --- Update History and Iteration Counter ---
        k++;
        // Update the history for the linear forecast of δx.
        delta_x_prev2 = delta_x_prev;
        delta_x_prev  = delta_x;

        // Check termination: if gradient is sufficiently small or max iterations reached.
        done = (gnorm <= gtol || k >= maxIter);
    }

    // Return the final estimate and the iteration count.
    return std::make_pair(x, k);
}

//------------------------------------------------------------------------------
// Example Usage
//------------------------------------------------------------------------------
int main() {
    // For demonstration, we work in a 3-dimensional space.
    const int dim = 3;

    // Define a sample data vector V.
    Eigen::VectorXd V(dim);
    V << 1.0, 2.0, 3.0;

    // Define a sample model function F(x).
    // In this simple example, we assume a linear model:
    //    F(x) = A * x + b.
    Eigen::MatrixXd A_model(dim, dim);
    A_model << 3, 0, 0,
               0, 2, 0,
               0, 0, 1;
    Eigen::VectorXd b_model(dim);
    b_model << 0.5, -0.5, 1.0;

    auto F = [=](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        return A_model * x + b_model;
    };

    // For a linear model the Jacobian is constant:
    auto dF = [=](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
        return A_model;
    };

    // Define the covariance matrix Sigma (here, the identity for simplicity).
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Identity(dim, dim);

    // Set the initial guess x0.
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);

    // Set algorithm parameters.
    int M = 5;               // Recompute the Jacobian every 5 iterations.
    int maxIter = 100;       // Maximum outer iterations.
    int maxIterCG = 50;      // Maximum inner CG iterations.
    double mu0 = 1.0;        // Initial damping.
    double gtol = 1e-5;      // Gradient tolerance.
    double tol_CG = 1e-5;    // Relative tolerance for CG.
    double atol_CG = 0.0;    // Absolute tolerance for CG.

    // Use the identity as the preconditioner (i.e. no preconditioning).
    auto identityPrecond = [](const Eigen::VectorXd &v) -> Eigen::VectorXd {
        return v;
    };

    // Run the flattened Levenberg–Marquardt algorithm.
    auto result = flattenedLM(V, F, dF, Sigma, x0, M, maxIter, maxIterCG,
                              mu0, gtol, tol_CG, atol_CG, identityPrecond);

    // Output the final result.
    std::cout << "Final x: " << result.first.transpose() << std::endl;
    std::cout << "Iterations: " << result.second << std::endl;

    return 0;
}
