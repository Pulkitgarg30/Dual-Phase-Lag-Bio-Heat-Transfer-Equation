import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# -----------------------
# Parameters (same as before)
# -----------------------
Foq = 0.5
Pf = 0.1
FoT = 0.01
theta_b = 0.0
Pm = 0.0
Pr0 = 1.0
a_gauss = 200.0
x_star = 0.5
A_bc = 1.0
B_bc = 1.0
g_func = lambda Fo: 0.0

# -----------------------
# Grid: use your reported values
# -----------------------
Nx = 201
Nt = 801
x_min, x_max = 0.0, 1.0
Fo_min, Fo_max = 0.0, 1.0

xs = np.linspace(x_min, x_max, Nx, dtype=np.float64)
Fos = np.linspace(Fo_min, Fo_max, Nt, dtype=np.float64)
dx = xs[1] - xs[0]
dFo = Fos[1] - Fos[0]
print("Grid: Nx, Nt =", Nx, Nt, " dx, dFo =", dx, dFo)

# -----------------------
# Source and N vector
# -----------------------
def Pr(x):
    return Pr0 * np.exp(-a_gauss*(x - x_star)**2)

def build_N_vector(Fo_val=None):
    # If N depends on Fo (g(Fo) in BC), evaluate accordingly.
    N = Pf**2 * theta_b + Pm + Pr(xs)
    # add BC-contribution if g != 0; for g=0 nothing extra
    if Fo_val is not None:
        gval = g_func(Fo_val)
        # last-entry BC term from paper's N (small correction) could be added here if needed.
    return N

# -----------------------
# Build M1 (same as before)
# -----------------------
def build_M1(Nx, dx, A_bc, B_bc):
    main = np.zeros(Nx, dtype=np.float64)
    lower = np.zeros(Nx-1, dtype=np.float64)
    upper = np.zeros(Nx-1, dtype=np.float64)
    for i in range(1, Nx-1):
        main[i] = -2.0/(dx*dx)
        lower[i-1] = 1.0/(dx*dx)
        upper[i] = 1.0/(dx*dx)
    main[0] = -2.0/(dx*dx)
    upper[0] = 2.0/(dx*dx)
    if A_bc == 0:
        main[-1] = -2.0/(dx*dx)
        lower[-1] = 1.0/(dx*dx)
    else:
        main[-1] = (-1.0 - (B_bc * dx / A_bc)) / (dx*dx)
        lower[-1] = 1.0/(dx*dx)
    M1 = sp.diags([lower, main, upper], offsets=[-1,0,1], format='csc')
    return M1

M1 = build_M1(Nx, dx, A_bc, B_bc)

# -----------------------
# Assemble M, C, K
# -----------------------
I = sp.eye(Nx, format='csc')
M_mat = Foq * I
C_mat = (1.0 + Foq * Pf**2) * I - FoT * M1
K_mat = Pf**2 * I - M1

# -----------------------
# Build block A matrix (constant)
# -----------------------
TL = sp.eye(Nx, format='csc')
TR = - (dFo/2.0) * sp.eye(Nx, format='csc')
BL = (dFo / (2.0 * Foq)) * K_mat
BR = sp.eye(Nx, format='csc') + (dFo / (2.0 * Foq)) * C_mat

A_block = sp.bmat([[TL, TR],
                   [BL, BR]], format='csc')

print("Factorizing CN block matrix...")
t0 = time.time()
A_fact = spla.factorized(A_block)
print("Factorized in {:.3f}s".format(time.time() - t0))

# -----------------------
# Prepare containers and initial conditions
# -----------------------
theta = np.zeros((Nt, Nx), dtype=np.float64)
u0 = np.zeros(Nx, dtype=np.float64)
v0 = np.zeros(Nx, dtype=np.float64)
theta[0, :] = u0.copy()
v_prev = v0.copy()

# -----------------------
# Compute consistent first step y^1 by solving CN block system from n=0->1
# Solve A * y1 = rhs where rhs uses u0,v0 and N0,N1
# -----------------------
print("Computing consistent first step (implicit CN for n=0->1)...")
N0 = build_N_vector(Fos[0])
N1 = build_N_vector(Fos[1])  # same in your problem, but keep general

# top RHS: u0 + (dFo/2)*v0
top_rhs = u0 + (dFo/2.0) * v0
# bottom RHS: v0 - (dFo/(2*Foq))*(K u0 + C v0) + (dFo/(2*Foq))*(N1 + N0)
bottom_rhs = v0 - (dFo / (2.0 * Foq)) * (K_mat.dot(u0) + C_mat.dot(v0)) + (dFo / (2.0 * Foq)) * (N1 + N0)
rhs1 = np.concatenate([top_rhs, bottom_rhs])

# diagnostics
print("RHS1 max/min:", np.max(rhs1), np.min(rhs1))
if not np.isfinite(rhs1).all():
    raise RuntimeError("Non-finite RHS for first implicit step; check N or initial conds.")

y1 = A_fact(rhs1)  # solve
u1 = y1[:Nx].copy()
v1 = y1[Nx:].copy()
theta[1, :] = u1.copy()
v_prev = v0.copy()
v_curr = v1.copy()

# quick safety check
if not np.isfinite(u1).all():
    raise RuntimeError("Non-finite u1 from implicit first step. Aborting.")

# -----------------------
# Time-marching loop (CN)
# -----------------------
print("Starting CN time stepping...")
tstart = time.time()
for n in range(1, Nt-1):
    u_n = theta[n, :].copy()
    v_n = v_curr.copy()
    # evaluate Nn and Nnp1
    Nn = build_N_vector(Fos[n])
    Nnp1 = build_N_vector(Fos[n+1])

    # build RHS parts
    top = u_n + (dFo/2.0) * v_n
    bottom = v_n - (dFo / (2.0 * Foq)) * (K_mat.dot(u_n) + C_mat.dot(v_n)) + (dFo / (2.0 * Foq)) * (Nnp1 + Nn)
    rhs = np.concatenate([top, bottom])

    # diagnostics: check rhs finite and magnitude before solve
    if not np.isfinite(rhs).all():
        print(f"Non-finite RHS at time n={n}, Fo={Fos[n]:.6e}")
        raise RuntimeError("Non-finite RHS during CN stepping.")

    if np.max(np.abs(rhs)) > 1e8:
        # warn but continue — this threshold is heuristic
        print(f"Large RHS magnitude at n={n}, max(rhs)={np.max(np.abs(rhs)):.3e}")

    y_np1 = A_fact(rhs)  # solve the block system
    u_np1 = y_np1[:Nx].copy()
    v_np1 = y_np1[Nx:].copy()

    # safety check
    if not np.isfinite(u_np1).all():
        print(f"Non-finite solution at step {n}, Fo={Fos[n]:.6e}. Aborting CN.")
        raise RuntimeError(f"Non-finite solution at step {n}")

    theta[n+1, :] = u_np1
    # advance velocities
    v_prev = v_n
    v_curr = v_np1

elapsed = time.time() - tstart
print("CN completed in {:.3f} s".format(elapsed))

Theta_cn = theta.copy()

# -----------------------
# plotting
# -----------------------
plt.figure(figsize=(6,4))
plt.pcolormesh(xs, Fos, Theta_cn, shading='auto')
plt.colorbar(label='theta')
plt.xlabel('x'); plt.ylabel('Fo'); plt.title('CN (full implicit) theta(x,Fo)')
plt.show()
