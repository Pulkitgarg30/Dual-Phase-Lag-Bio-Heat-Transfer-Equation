import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time

np.random.seed(1234)
tf.set_random_seed(1234)

Foq = 0.5        # F_{oq}
Pf = 0.1         # P_f
FoT = 0.01       # Fo_T
theta_b = 0.0    # background temp (dimensionless)
Pm = 0.0         # metabolic term
Pr0 = 1.0        # source amplitude
a_gauss = 200.0  # gaussian width
x_star = 0.5     # tumor location in [0,1]
A_bc = 1.0       # Robin A
B_bc = 1.0       # Robin B

Fo_max = 1.0     # max Fo

N_coll = 5000
N_bc = 300
N_ic = 200

layers = [2, 80, 80, 80, 80, 1]

# ----------------------
# Utilities: NN init & forward
# ----------------------
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=stddev), dtype=tf.float32)

def neural_net(X, weights, biases):
    H = 2.0*(X - X_lb) / (X_ub - X_lb) - 1.0
    num_layers = len(weights) + 1
    for l in range(num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

# ----------------------
# Build TF graph
# ----------------------
x_tf     = tf.placeholder(tf.float32, shape=[None, 1])
Fo_tf    = tf.placeholder(tf.float32, shape=[None, 1])
Fo_bc_tf = tf.placeholder(tf.float32, shape=[None, 1])

X_lb = tf.placeholder(tf.float32, shape=[2])
X_ub = tf.placeholder(tf.float32, shape=[2])

# NN weights & biases
weights = []
biases  = []
for l in range(len(layers)-1):
    W = xavier_init([layers[l], layers[l+1]])
    b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
    weights.append(W)
    biases.append(b)

# ----------------------
# Trial solution & derivatives
# ----------------------
X_coll     = tf.concat([x_tf, Fo_tf], axis=1)
N_out      = neural_net(X_coll, weights, biases)
theta_coll = tf.multiply(tf.square(Fo_tf), N_out)

theta_x    = tf.gradients(theta_coll, x_tf)[0]
theta_Fo   = tf.gradients(theta_coll, Fo_tf)[0]
theta_xx   = tf.gradients(theta_x,    x_tf)[0]
theta_FoFo = tf.gradients(theta_Fo,   Fo_tf)[0]
theta_xFo  = tf.gradients(theta_x,    Fo_tf)[0]

# source term
Pr_coll = Pr0 * tf.exp(-a_gauss * tf.square(x_tf - x_star))

# ----------------------
# PDE residual R
# ----------------------
R_coll = (Foq * theta_FoFo
          + (1.0 + Foq * Pf**2) * theta_Fo
          + (Pf**2) * theta_coll
          - theta_xx
          - FoT * theta_xFo
          - (Pf**2 * theta_b + Pm + Pr_coll)
         )

# ==============================================================
#  gPINN: gradient of residual w.r.t. x and Fo
#  For the exact solution R = 0 everywhere, so
#  dR/dx = 0 and dR/dFo = 0 must also hold
# ==============================================================
R_x  = tf.gradients(R_coll, x_tf)[0]    # dR/dx
R_Fo = tf.gradients(R_coll, Fo_tf)[0]   # dR/dFo

# ----------------------
# Boundary residuals (identical to base code)
# ----------------------
# symmetry at x=0: dtheta/dx = 0
x0            = tf.zeros_like(Fo_bc_tf)
X_bc0         = tf.concat([x0, Fo_bc_tf], axis=1)
N_bc0         = neural_net(X_bc0, weights, biases)
theta_bc0     = tf.multiply(tf.square(Fo_bc_tf), N_bc0)
dtheta_dx_bc0 = tf.gradients(theta_bc0, x0)[0]

# Robin at x=1: A*dtheta/dx + B*theta = 0
x1            = tf.ones_like(Fo_bc_tf)
X_bc1         = tf.concat([x1, Fo_bc_tf], axis=1)
N_bc1         = neural_net(X_bc1, weights, biases)
theta_bc1     = tf.multiply(tf.square(Fo_bc_tf), N_bc1)
dtheta_dx_bc1 = tf.gradients(theta_bc1, x1)[0]
g_Fo          = tf.zeros_like(Fo_bc_tf)
robin_res     = A_bc * dtheta_dx_bc1 + B_bc * theta_bc1 - g_Fo

# ----------------------
# Loss components
# ----------------------
loss_pde = tf.reduce_mean(tf.square(R_coll))
loss_bc  = tf.reduce_mean(tf.square(dtheta_dx_bc0)) + tf.reduce_mean(tf.square(robin_res))

# ==============================================================
#  gPINN gradient loss
#  dR/dx and dR/dFo should both be 0 for the exact solution
# ==============================================================
loss_g = tf.reduce_mean(tf.square(R_x)) + tf.reduce_mean(tf.square(R_Fo))

# gPINN total loss — gradient term weighted by w_g = 0.1
w_g  = 0.1
loss = loss_pde + loss_bc + w_g * loss_g

# ----------------------
# Optimisers
# ----------------------
optimizer_adam = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op_adam  = optimizer_adam.minimize(loss)

optimizer_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(
    loss,
    method='L-BFGS-B',
    options={'maxiter': 50000,
             'maxfun':  50000,
             'maxcor':  50,
             'maxls':   50,
             'ftol':    1.0 * np.finfo(float).eps})

# ----------------------
# Session & init
# ----------------------
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False))
sess.run(tf.global_variables_initializer())

# ----------------------
# Sampling helpers
# ----------------------
def sample_collocation(N):
    x  = np.random.rand(N, 1).astype(np.float32)
    Fo = (np.random.rand(N, 1) * Fo_max).astype(np.float32)
    return x, Fo

def sample_bc(N):
    return (np.random.rand(N, 1) * Fo_max).astype(np.float32)

# ----------------------
# Training — Adam phase
# ----------------------
def train_adam(n_iters=8000, batch_coll=4000, batch_bc=200):
    t0 = time.time()
    for it in range(n_iters):
        xr, Forr = sample_collocation(batch_coll)
        Fobc     = sample_bc(batch_bc)

        feed = {x_tf:     xr,
                Fo_tf:    Forr,
                Fo_bc_tf: Fobc,
                X_lb: np.array([0.0, 0.0],    dtype=np.float32),
                X_ub: np.array([1.0, Fo_max], dtype=np.float32)}

        sess.run(train_op_adam, feed_dict=feed)

        if it % 500 == 0:
            l_tot, l_pde, l_bc, l_g = sess.run(
                [loss, loss_pde, loss_bc, loss_g], feed_dict=feed)
            print("Adam iter {:6d} | loss {:.3e} | pde {:.3e} | bc {:.3e} "
                  "| grad {:.3e} | time {:.2f}s".format(
                  it, l_tot, l_pde, l_bc, l_g, time.time()-t0))
            t0 = time.time()

# ----------------------
# Training — L-BFGS phase
# ----------------------
def train_lbfgs(n_coll=6000, n_bc=400):
    xr, Forr = sample_collocation(n_coll)
    Fobc     = sample_bc(n_bc)
    feed = {x_tf:     xr,
            Fo_tf:    Forr,
            Fo_bc_tf: Fobc,
            X_lb: np.array([0.0, 0.0],    dtype=np.float32),
            X_ub: np.array([1.0, Fo_max], dtype=np.float32)}
    print("Starting L-BFGS...")
    optimizer_lbfgs.minimize(sess, feed_dict=feed)
    print("L-BFGS finished.")

# ----------------------
# Run training & visualize
# ----------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  gPINN  Bioheat Equation  —  Adam 8000 iters")
    print("=" * 65)

    t_total_start = time.time()          # ← total timer starts here

    train_adam(n_iters=8000, batch_coll=4000, batch_bc=200)

    t_lbfgs_start = time.time()          # ← L-BFGS timer starts here
    print("Adam finished. Now L-BFGS...")
    train_lbfgs(n_coll=6000, n_bc=400)
    lbfgs_time = time.time() - t_lbfgs_start

    total_time = time.time() - t_total_start
    print(f"  Adam time  : {total_time - lbfgs_time:.1f}s")
    print(f"  L-BFGS time: {lbfgs_time:.1f}s")
    print(f"  Total time : {total_time:.1f}s")
    np.save("gpinn_bioheat_time.npy", np.array([total_time]))

    print("Training complete. Evaluating solution...")

    # ----------------------
    # Evaluation on 200x200 grid
    # ----------------------
    Nx = 200; Nt = 200
    xs  = np.linspace(0.0, 1.0,    Nx, dtype=np.float32)
    Fos = np.linspace(0.0, Fo_max, Nt, dtype=np.float32)
    Xg, Tg  = np.meshgrid(xs, Fos, indexing='xy')
    x_flat  = Xg.reshape(-1, 1).astype(np.float32)
    Fo_flat = Tg.reshape(-1, 1).astype(np.float32)

    feed_eval = {x_tf:  x_flat,
                 Fo_tf: Fo_flat,
                 X_lb:  np.array([0.0, 0.0],    dtype=np.float32),
                 X_ub:  np.array([1.0, Fo_max], dtype=np.float32)}

    theta_flat = sess.run(theta_coll, feed_dict=feed_eval)
    Theta      = theta_flat.reshape(Nt, Nx)

    # ----------------------
    # Plots
    # ----------------------
    idx = np.argmin(np.abs(xs - x_star))
    plt.figure(figsize=(6, 4))
    plt.plot(Fos, Theta[:, idx], label=f"x={xs[idx]:.3f}")
    plt.xlabel("Fo"); plt.ylabel("theta")
    plt.title("gPINN: theta(Fo) at x*")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("gpinn_bioheat_slice.png", dpi=150)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(xs, Fos, Theta, shading='auto')
    plt.xlabel("x"); plt.ylabel("Fo")
    plt.title("gPINN: theta(x, Fo)")
    plt.colorbar(); plt.tight_layout()
    plt.savefig("gpinn_bioheat_heatmap.png", dpi=150)
    plt.show()

    # ----------------------
    # Save solution for comparison
    # ----------------------
    np.save("gpinn_bioheat.npy", Theta)
    print("Saved: gpinn_bioheat.npy")