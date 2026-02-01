import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# -----------------------
# Problem parameters (match your PINN/CN runs)
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
g_func = lambda Fo: 0.0   # Robin boundary g(Fo), zero here

# -----------------------
# Domain & grid (choose or match existing)
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
# Source function and vector N(Fo) (space part)
# -----------------------
def Pr(x):
    return Pr0 * np.exp(-a_gauss*(x - x_star)**2)

def build_N_vector(Fo_val=None):
    # space-only part; if g(Fo) contributes, include here
    N = Pf**2 * theta_b + Pm + Pr(xs)
    # If Robin g != 0 you may need to add a BC-dependent entry in N here.
    return N

# -----------------------
# Build M1 (discrete ∂²/∂x²) with Neumann at left and Robin at right
# M1 * u ~ d^2 u / dx^2
# -----------------------
def build_M1(Nx, dx, A_bc, B_bc):
    main = np.zeros(Nx, dtype=np.float64)
    lower = np.zeros(Nx-1, dtype=np.float64)
    upper = np.zeros(Nx-1, dtype=np.float64)

    for i in range(1, Nx-1):
        main[i]  = -2.0/(dx*dx)
        lower[i-1] = 1.0/(dx*dx)
        upper[i] = 1.0/(dx*dx)

    # left (i=0) symmetry: dθ/dx=0 -> mirror -> Dxx_0 = (θ1 - 2θ0 + θ1)/dx^2
    main[0] = -2.0/(dx*dx)
    upper[0] = 2.0/(dx*dx)

    # right (i = Nx-1)
    if A_bc == 0:
        # Dirichlet: θ_N = 0
        main[-1] = -2.0/(dx*dx)
        lower[-1] = 1.0/(dx*dx)
    else:
        # ghost elimination: θ_ghost = θ_{N-1} + (dx/A)*(g - B θ_{N-1})
        # with g=0 => θ_ghost = θ_{N-1}*(1 - B dx/A)
        main[-1]  = (-1.0 - (B_bc * dx / A_bc)) / (dx*dx)
        lower[-1] = 1.0/(dx*dx)

    M1 = sp.diags([lower, main, upper], offsets=[-1,0,1], format='csc')
    return M1

M1 = build_M1(Nx, dx, A_bc, B_bc)

# -----------------------
# Assemble M, C, K (matrix form M u'' + C u' + K u = N)
# Note sign conventions chosen so diffusion is stabilizing:
#   M = Foq * I
#   C = (1 + Foq * Pf^2) * I - FoT * M1   (FoT term subtractive)
#   K = Pf^2 * I - M1
# -----------------------
I = sp.eye(Nx, format='csc')
M_mat = Foq * I
C_mat = (1.0 + Foq * Pf**2) * I - FoT * M1
K_mat = Pf**2 * I - M1

# -----------------------
# Newmark parameters (recommended)
# -----------------------
beta  = 1.0/4.0
gamma = 1.0/2.0

# -----------------------
# Precompute effective stiffness K_eff (sparse)
# K_eff = K + (gamma/(beta*dt)) C + (1/(beta*dt^2)) M
# We'll factorize K_eff once (sparse) for speed.
# -----------------------
coeff_C = (gamma / (beta * dFo))
coeff_M = (1.0 / (beta * dFo * dFo))

K_eff = (K_mat + coeff_C * C_mat + coeff_M * M_mat).tocsc()
print("Factorizing Newmark effective stiffness (size {})...".format(K_eff.shape))
t0 = time.time()
Keff_solver = spla.factorized(K_eff)   # returns solver fn
print("Factorized in {:.3f}s".format(time.time() - t0))

# -----------------------
# Containers & initial conditions
# -----------------------
Theta_newmark = np.zeros((Nt, Nx), dtype=np.float64)   # solution u at each Fo
u_n = np.zeros(Nx, dtype=np.float64)   # u^n
v_n = np.zeros(Nx, dtype=np.float64)   # v^n = u'
# initial acceleration a0 from equation: M a0 = N0 - C v0 - K u0
N0 = build_N_vector(Fos[0])
# Since u0 = 0 and v0 = 0 (IC in paper), a0 = M^{-1} * N0
a_n = (M_mat.dot(np.zeros(Nx)) * 0.0)  # placeholder
# compute properly:
# M a0 = N0 - C v0 - K u0  => a0 = M^{-1} N0  (since u0=v0=0)
a_n = spla.spsolve(M_mat, N0)   # M is scalar*I so quick

# store initial
Theta_newmark[0, :] = u_n.copy()

# If you want a consistent u1 you can do an initial implicit step (but Newmark formula handles it)
# We'll compute u1 via Newmark by solving the usual K_eff system for n=0 -> 1.
# For that we need f_eff at n=1 (N1). We'll compute in the loop as general case.

# -----------------------
# Time stepping loop
# -----------------------
print("Starting Newmark time march (β=1/4, γ=1/2)...")
t_start = time.time()
for n in range(0, Nt-1):
    # Evaluate N at n and n+1 (N depends only on x here, but kept general)
    Nn = build_N_vector(Fos[n])
    Nnp1 = build_N_vector(Fos[n+1])

    # Build effective right-hand side f_eff:
    # f_eff = N_{n+1} + M*( (1/(beta dt^2)) u_n + (1/(beta dt)) v_n + (1/(2beta)-1) a_n )
    #         + C*( (gamma/(beta dt)) u_n + (gamma/beta - 1) v_n + dt*(gamma/(2beta) - 1) a_n )
    term_M = ( (1.0/(beta * dFo*dFo)) * u_n
               + (1.0/(beta * dFo)) * v_n
               + ( (1.0/(2.0*beta)) - 1.0) * a_n )
    term_C = ( (gamma/(beta * dFo)) * u_n
               + ( (gamma / beta) - 1.0 ) * v_n
               + dFo * ( (gamma/(2.0*beta)) - 1.0 ) * a_n )

    rhs = Nnp1 + (M_mat.dot(term_M)) + (C_mat.dot(term_C))

    # Solve for u_{n+1}: K_eff * u_{n+1} = rhs
    u_np1 = Keff_solver(rhs)

    # compute a_{n+1} from displacement formula:
    a_np1 = (u_np1 - u_n - dFo * v_n - dFo*dFo * (0.5 - beta) * a_n) / (beta * dFo*dFo)

    # compute v_{n+1}
    v_np1 = v_n + dFo * ( (1.0 - gamma) * a_n + gamma * a_np1 )

    # store and shift
    Theta_newmark[n+1, :] = u_np1.copy()
    u_n = u_np1
    v_n = v_np1
    a_n = a_np1

    # safety check
    if not np.isfinite(u_n).all():
        raise RuntimeError(f"Non-finite values at step {n+1}, Fo={Fos[n+1]:.6f}")

elapsed = time.time() - t_start
print("Newmark march completed in {:.3f}s".format(elapsed))

# -----------------------
# Plots (similar to earlier)
# -----------------------
plt.figure(figsize=(6,4))
plt.pcolormesh(xs, Fos, Theta_newmark, shading='auto')
plt.colorbar(label='theta')
plt.xlabel('x'); plt.ylabel('Fo'); plt.title('Newmark-β (1/4,1/2): theta(x,Fo)')
plt.show()

# theta(Fo) at slices
x_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
plt.figure(figsize=(6,4))
for xv in x_vals:
    idx = np.argmin(np.abs(xs - xv))
    plt.plot(Fos, Theta_newmark[:, idx], label=f"x={xv:.2f}")
plt.xlabel('Fo'); plt.ylabel('theta'); plt.title('Newmark: theta(Fo) at x slices')
plt.legend(); plt.grid(True); plt.show()

# optional: compare with PINN 'Theta' if present (assumes same grid)
try:
    Theta  # PINN result variable used earlier in your notebook
    if Theta.shape == Theta_newmark.shape:
        diff = Theta_newmark - Theta
        L2_rel = np.linalg.norm(diff) / (np.linalg.norm(Theta_newmark) + 1e-16)
        Linf_rel = np.max(np.abs(diff)) / (np.max(np.abs(Theta_newmark)) + 1e-16)
        print(f"Newmark vs PINN: Relative L2 = {L2_rel:.3e}, Relative Linf = {Linf_rel:.3e}")
        plt.figure(figsize=(6,4))
        plt.pcolormesh(xs, Fos, np.abs(diff), shading='auto')
        plt.colorbar(label='|Newmark - PINN|')
        plt.title('Absolute difference Newmark - PINN')
        plt.xlabel('x'); plt.ylabel('Fo'); plt.show()
    else:
        print("PINN Theta exists but shape mismatch - interpolate before comparing.")
except NameError:
    print("No PINN variable 'Theta' found in namespace - skip comparison.")
