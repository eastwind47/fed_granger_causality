import os
import numpy as np
from scipy.linalg import block_diag, eigvals
import matplotlib.pyplot as plt

def make_positive_block(P, rng, eps=1e-3):
    """
    Return a P×P matrix with strictly positive entries in (eps, 1].
    """
    A = rng.rand(P, P)
    return A * (1 - eps) + eps

def generate_lti_data(
    N: int,
    D: int,
    P: int,
    T: int,
    R,
    Q=None,
    u=None,
    B=None,
    seed=None,
    dependencies=None
):
    """
    Generate LTI state-space data:
        x_{t+1} = A x_t + B u_t + w_t
        y_t     = C x_t     + v_t
    """
    rng = np.random.RandomState(seed)
    n_x, n_y = N*P, N*D

    # build A
    A_blocks = [make_positive_block(P, rng) for _ in range(N)]
    A = block_diag(*A_blocks)
    if dependencies:
        for src, dst in dependencies:
            A[dst*P:(dst+1)*P, src*P:(src+1)*P] = make_positive_block(P, rng)
    rho = max(abs(eigvals(A)))
    A /= (1.1 * rho)

    # build C
    C = block_diag(*[rng.randn(D, P) for _ in range(N)])

    # Q covariance
    if Q is None:
        Qmat = np.zeros((n_x, n_x))
    else:
        Q = np.array(Q)
        if Q.ndim == 0:
            Qmat = Q * np.eye(n_x)
        elif Q.ndim == 1:
            Qmat = np.diag(Q)
        else:
            Qmat = Q

    # R covariance
    if isinstance(R, (list, tuple)):
        R_blocks = []
        for Ri in R:
            Ri = np.array(Ri)
            if Ri.ndim == 0:
                R_blocks.append(Ri * np.eye(D))
            elif Ri.ndim == 1:
                R_blocks.append(np.diag(Ri))
            else:
                R_blocks.append(Ri)
        Rmat = block_diag(*R_blocks)
    else:
        R = np.array(R)
        if R.ndim == 0:
            Rmat = R * np.eye(n_y)
        elif R.ndim == 1:
            Rmat = np.diag(R)
        else:
            Rmat = R

    # inputs
    if u is None:
        u = np.zeros((T, 0))
    if B is None:
        B = np.zeros((n_x, u.shape[1]))

    # allocate & sample noise
    x = np.zeros((T, n_x))
    y = np.zeros((T, n_y))
    w = rng.multivariate_normal(np.zeros(n_x), Qmat, size=T)
    v = rng.multivariate_normal(np.zeros(n_y), Rmat, size=T)

    # simulate
    for t in range(T-1):
        x[t+1] = A @ x[t] + B @ u[t] + w[t]
        y[t]   = C @ x[t] + v[t]
    y[-1] = C @ x[-1] + v[-1]

    return x, y, A, C, B

if __name__ == "__main__":
    # parameters
    N, D, P, T = 3, 8, 2, 10000
    R = [
        0.0001 * np.eye(D),
        0.0003 * np.eye(D),
        0.03   * np.eye(D)
    ]
    Q = 1e-5
    deps = [(0,1), (2,1)]
    windows = [(1000,2000,1.0), (3000,5000,-0.5), (7000,9000,2.0)]

    # build u
    n_x = N * P
    u = np.zeros((T, n_x))
    for t0, t1, mag in windows:
        u[t0:t1, :] = mag
    B = 0.01 * np.eye(n_x)

    # generate data
    x, y, A_true, C, B = generate_lti_data(
        N=N, D=D, P=P, T=T,
        R=R, Q=Q,
        u=u, B=B,
        seed=42,
        dependencies=deps
    )

    # directory structure
    base_dir = f"{N}_components"
    os.makedirs(base_dir, exist_ok=True)

    # save full A
    np.savetxt(os.path.join(base_dir, "A_true.csv"), A_true, delimiter=",")

    # per-component files
    for i in range(N):
        comp_dir = os.path.join(base_dir, f"C{i+1}")
        os.makedirs(comp_dir, exist_ok=True)

        # slice data
        xi = x[:, i*P:(i+1)*P]
        yi = y[:, i*D:(i+1)*D]
        Ai = A_true[i*P:(i+1)*P, i*P:(i+1)*P]
        Ci = C[i*D:(i+1)*D, i*P:(i+1)*P]

        # save CSVs
        np.savetxt(os.path.join(comp_dir, "x.csv"), xi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "y.csv"), yi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "A.csv"), Ai, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "C.csv"), Ci, delimiter=",")