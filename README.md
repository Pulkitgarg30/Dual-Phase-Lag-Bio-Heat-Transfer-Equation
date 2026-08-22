# Dual-Phase-Lag Bio-Heat Transfer Modeling using Physics-Informed Neural Networks (PINN)

## 📌 Project Overview

This project develops a **Physics-Informed Neural Network (PINN) framework** for solving the **Dual-Phase-Lag (DPL) bio-heat transfer equation**, which extends the classical Pennes bio-heat equation to capture finite thermal wave propagation and micro-scale lag effects. The DPL model is particularly suited for fast transient thermal processes such as **hyperthermia cancer treatment**.

The study implements and compares **four PINN variants**:

1. **Standard PINN** — the baseline framework of Raissi et al. (2019)
2. **SA-PINN** — Self-Adaptive PINN with adversarial loss weighting (McClenny & Braga-Neto, 2023)
3. **gPINN** — Gradient-enhanced PINN with residual smoothness constraints (Yu et al., 2022)
4. **SA-gPINN** — a **novel hybrid proposed in this work**, combining self-adaptive weighting with gradient enhancement

All four PINN variants are validated against two **independent classical numerical schemes** that serve as trusted references:

- **Crank–Nicolson (CN)** finite difference scheme
- **Newmark-β** time integration method

The four PINN variants are benchmarked on **two heat-source scenarios** — a sharp Gaussian source (the primary hyperthermia case) and a smooth polynomial source (a controlled validation case) — to identify when each variant excels.

---

## 🎯 Motivation

Thermal therapies such as hyperthermia require precise prediction and control of temperature distribution inside biological tissue. Classical models based on Fourier's law assume **infinite speed of heat propagation**, which is physically unrealistic for:

- Rapid heating processes
- Micro-scale tissue interactions
- Short-time thermal responses

The **Dual-Phase-Lag (DPL) model** resolves these limitations by introducing:

- A **phase lag of heat flux** (τ_q)
- A **phase lag of temperature gradient** (τ_T)

These additions allow the model to capture **wave-like thermal behavior** and provide more accurate predictions for biomedical applications.

---

## 📐 Mathematical Formulation

### Dual-Phase-Lag Bio-Heat Equation (Dimensionless Form)

The governing equation solved in this project is:

$$Fo_q \frac{\partial^2 \theta}{\partial Fo^2} + (1 + Fo_q P_f^2)\frac{\partial \theta}{\partial Fo} + P_f^2 \theta = \frac{\partial^2 \theta}{\partial x^2} + Fo_T \frac{\partial^3 \theta}{\partial x^2 \partial Fo} + P_f^2 \theta_b + P_m + P_r(x)$$

Where:
- $\theta$ is the dimensionless temperature
- $Fo$ is the Fourier number (dimensionless time)
- $x$ is the dimensionless spatial coordinate
- $Fo_q$ represents the heat-flux phase lag
- $Fo_T$ represents the temperature-gradient phase lag
- $P_f$ is the blood perfusion parameter
- $P_r(x)$ is the applied heat source

### Heat Source Profiles

Two heat sources are studied to probe the behavior of the four PINN variants:

**(a) Sharp Gaussian source (primary case — hyperthermia model):**

$$P_r(x) = P_{r0} \exp\left(-a(x - x^*)^2\right), \quad a = 200, \quad x^* = 0.5$$

Models a localized tumor heating region. Sharp derivative: $\|\partial P_r/\partial x\|_\infty \approx 12.13$.

**(b) Smooth polynomial source (validation case):**

$$P_r(x) = P_{r0}\, x(1-x), \quad P_{r0} = 1$$

Smooth, bounded derivative: $\|\partial P_r/\partial x\|_\infty = 1$.

The two profiles differ by a **12× factor in derivative magnitude** — equivalently, ~147× in the squared quantity that enters the gradient-residual loss of gPINN and SA-gPINN. This contrast is central to the mechanistic findings of this study.

---

## 🧪 Initial and Boundary Conditions

**Initial conditions:**
- $\theta(x, 0) = 0$
- $\frac{\partial \theta}{\partial Fo}(x, 0) = 0$

**Boundary conditions:**
- **Symmetry (Neumann)** at tumor center: $\frac{\partial \theta}{\partial x} = 0$ at $x = 0$
- **Robin (third-kind)** at outer boundary: $A\frac{\partial \theta}{\partial x} + B\theta = 0$ at $x = 1$

---

## 🤖 The Four PINN Variants

All four variants share the same fully-connected neural network architecture ([2, 80, 80, 80, 80, 1] with tanh activations) and the same **trial-solution ansatz**:

$$\theta_{NN}(x, Fo) = Fo^2 \cdot \mathcal{N}(x, Fo; w)$$

This ansatz **hard-codes both initial conditions exactly** — regardless of the network parameters — eliminating the IC loss term entirely and reducing the training problem to two competing objectives (PDE residual and boundary-condition residual).

### 1. Standard PINN
Baseline framework with unweighted composite loss:
$$\mathcal{L}_{PINN} = \mathcal{L}_r + \mathcal{L}_{BC}$$

### 2. SA-PINN (Self-Adaptive PINN)
Introduces learnable adversarial weights that dynamically rebalance loss terms:
$$\mathcal{L}_{SA-PINN} = \lambda_r \mathcal{L}_r + \lambda_{BC} \mathcal{L}_{BC}$$
Weights are trained by adversarial ascent while the network minimizes.

### 3. gPINN (Gradient-Enhanced PINN)
Adds a gradient-residual loss enforcing $\partial R/\partial x \equiv 0$ and $\partial R/\partial Fo \equiv 0$:
$$\mathcal{L}_{gPINN} = \mathcal{L}_r + \mathcal{L}_{BC} + w_g \mathcal{L}_g$$
Requires third-order automatic differentiation.

### 4. SA-gPINN (Proposed — Novel Contribution) ⭐
Combines self-adaptive weighting with gradient enhancement:
$$\mathcal{L}_{SA-gPINN} = \lambda_r \mathcal{L}_r + \lambda_{BC} \mathcal{L}_{BC} + \lambda_g \mathcal{L}_g$$
Three-weight adversarial saddle-point optimization. Predicted to excel when the source is smooth enough that $\mathcal{L}_g$ carries genuine physics rather than source-derivative noise.

---

## 🧮 Classical Reference Schemes

### Crank–Nicolson (CN)
- Reduces the DPL system to a first-order state-space form
- Applies trapezoidal integration → constant block linear system
- LU-factorized once at initialization; only substitutions per step
- **Second-order accurate, unconditionally stable, ~2.7 seconds total runtime**

### Newmark-β
- Operates directly on the second-order form (no state-space reduction)
- Parameters: $\beta = 1/4$, $\gamma = 1/2$ (constant-average-acceleration)
- Constant effective-stiffness matrix, factorized once
- **Second-order accurate, unconditionally stable, ~1.0 second total runtime**

**Cross-scheme validation:** CN and Newmark-β agree to **machine precision (~10⁻¹²)** on both source profiles, providing a bitwise-trusted reference.

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---|---|
| Adam iterations | 8000 |
| Adam learning rate | 10⁻³ |
| SA weight learning rate | 10⁻² |
| L-BFGS-B tolerance | machine ε |
| Collocation points (Adam) | 4000 (resampled per iter) |
| BC points (Adam) | 200 (resampled per iter) |
| Random seed | 1234 |

Training pipeline: **Adam (exploration) → L-BFGS-B (fine-tuning)**.

---

## 📊 Results — Case 1: Gaussian Heat Source (Primary Case)

This is the **primary hyperthermia scenario** with sharp Gaussian source ($a = 200$).

### PDE Residual and Boundary Condition Residuals

| Method | PDE Residual RMS | BC x=0 (Neumann) | BC x=1 (Robin) |
|---|---|---|---|
| PINN | 2.2175 × 10⁻³ | 1.2670 × 10⁻³ | 1.5335 × 10⁻³ |
| **SA-PINN** | **1.8336 × 10⁻³** | **1.0821 × 10⁻³** | **1.0265 × 10⁻³** |
| gPINN | 1.9667 × 10⁻³ | 3.7886 × 10⁻³ | 3.9385 × 10⁻³ |
| SA-gPINN | 1.8773 × 10⁻³ | 3.9908 × 10⁻³ | 4.1451 × 10⁻³ |
| *CN (reference)* | *1.4197 × 10⁻²\** | *3.8633 × 10⁻⁴* | *6.2378 × 10⁻⁴* |
| *Newmark (reference)* | *1.4197 × 10⁻²\** | *3.8633 × 10⁻⁴* | *6.2378 × 10⁻⁴* |

*\*CN/Newmark PDE residual is inflated by np.gradient truncation error, not solution error.*

### Solution Difference from CN Reference

| Method | Max \|θ − θ_CN\| | Mean \|θ − θ_CN\| |
|---|---|---|
| PINN | 1.2090 × 10⁻³ | 2.9247 × 10⁻⁴ |
| **SA-PINN** | **1.2058 × 10⁻³** | 3.0067 × 10⁻⁴ |
| gPINN | 1.1562 × 10⁻³ | 3.4482 × 10⁻⁴ |
| SA-gPINN | 1.2070 × 10⁻³ | 3.8254 × 10⁻⁴ |
| *Newmark* | *1.17 × 10⁻¹²* | *2.44 × 10⁻¹³* |

### 🏆 Finding on Gaussian Source

**SA-PINN wins on all three metrics** (PDE residual, both BC residuals). The gradient-enhanced methods (gPINN, SA-gPINN) show a **~4× BC-residual penalty** driven by source-derivative corruption of $\mathcal{L}_g$: the sharp Gaussian generates a large $\|\partial P_r/\partial x\|_\infty \approx 12.13$ that dominates the gradient loss and starves $\lambda_{BC}$ of gradient budget.

---

## 📊 Results — Case 2: Polynomial Heat Source (Validation Case)

This is the **controlled validation case** with smooth polynomial source $P_r(x) = x(1-x)$, designed to test the source-derivative mechanism hypothesis.

### PDE Residual and Boundary Condition Residuals

| Method | PDE Residual RMS | BC x=0 (Neumann) | BC x=1 (Robin) |
|---|---|---|---|
| PINN | 1.1196 × 10⁻² | 8.4808 × 10⁻⁴ | 1.1170 × 10⁻³ |
| SA-PINN | 1.1086 × 10⁻² | **8.0397 × 10⁻⁴** | 1.0025 × 10⁻³ |
| gPINN | 1.1018 × 10⁻² | 1.4299 × 10⁻³ | 9.2685 × 10⁻⁴ |
| **SA-gPINN** ⭐ | **1.1000 × 10⁻²** | 1.1335 × 10⁻³ | **9.0508 × 10⁻⁴** |
| *CN (reference)* | *9.2006 × 10⁻³* | *4.0235 × 10⁻⁴* | *5.5038 × 10⁻⁴* |
| *Newmark (reference)* | *9.2006 × 10⁻³* | *4.0235 × 10⁻⁴* | *5.5038 × 10⁻⁴* |

### Solution Difference from CN Reference

| Method | Max \|θ − θ_CN\| | Mean \|θ − θ_CN\| |
|---|---|---|
| PINN | 5.5897 × 10⁻⁴ | 1.5666 × 10⁻⁴ |
| **SA-PINN** | **4.6749 × 10⁻⁴** | **1.4276 × 10⁻⁴** |
| gPINN | 6.9888 × 10⁻⁴ | 1.9557 × 10⁻⁴ |
| SA-gPINN | 5.4767 × 10⁻⁴ | 1.6820 × 10⁻⁴ |
| *Newmark* | *1.53 × 10⁻¹²* | *3.13 × 10⁻¹³* |

### 🏆 Finding on Polynomial Source

**SA-gPINN attains the lowest PDE residual** and **wins the Robin BC at x = 1**. Critically, the **~4× BC penalty observed on the Gaussian source collapses to comparability with SA-PINN**: SA-gPINN's BC residuals fall from ~4 × 10⁻³ (Gaussian) to ~1 × 10⁻³ (polynomial), directly matching the theoretical prediction.

---

## 🔬 Key Finding: The Source-Derivative Mechanism

The two heat-source scenarios together validate a novel mechanistic prediction:

> **Gradient-enhanced PINN variants are compromised when the source term has sharp spatial features (because $\mathcal{L}_g$ becomes dominated by source-derivative noise), and become competitive with — or superior to — their non-gradient counterparts when the source is smooth.**

The BC-penalty collapse from ~4× (Gaussian) to ~1× (polynomial), as source-derivative magnitude drops from 12.13 to 1.0, is direct empirical confirmation. This provides a **problem-specific selection criterion** for PINN methodology:

- **Sharp-source problems** → prefer SA-PINN
- **Smooth-source problems** → SA-gPINN offers additional benefit

---

## ⏱ Wall-Clock Training Cost

| Method | Total training time |
|---|---|
| PINN | ~51 min |
| SA-PINN | ~67 min |
| gPINN | ~5.2 hr |
| SA-gPINN | ~10.7 hr |

Cost scales with the **order of automatic differentiation** (gPINN, SA-gPINN require third-order) and the **adversarial update overhead** (SA-PINN, SA-gPINN).

---

## ✅ Boundary Condition Verification

- Symmetry and Robin boundary conditions satisfied within O(10⁻³) for SA-PINN on Gaussian source
- Residual analysis confirms strong physical consistency across all four PINN variants
- No explicit penalty tuning required — SA-PINN and SA-gPINN adapt weights automatically

---

## 🛠 Technologies Used

- **Python 3**
- **TensorFlow** — neural network implementation and automatic differentiation
- **NumPy / SciPy** — classical numerical schemes and post-processing
- **Matplotlib** — visualization and heatmaps
- **Finite Difference Methods** — Crank–Nicolson and Newmark-β reference solvers

---

## 🚀 Key Contributions

1. **Proposed SA-gPINN**, a novel hybrid combining self-adaptive weighting with gradient enhancement, and derived its adversarial three-weight loss formulation.
2. **Systematically compared four PINN variants** (PINN, SA-PINN, gPINN, SA-gPINN) on the DPL bio-heat equation for the first time.
3. **Identified and validated a source-derivative failure mechanism** predicting when gradient-enhanced PINNs succeed vs. fail.
4. **Validated all PINN variants against two mathematically independent classical schemes** (CN and Newmark-β) agreeing to machine precision.
5. **Provided a design principle** for practitioner choice between SA-PINN and SA-gPINN based on source smoothness.

---

## 🔮 Future Scope

- Extension to **2D and 3D patient-specific tissue geometries**
- **Inverse PINN** for parameter estimation of unknown phase-lag coefficients (τ_q, τ_T)
- **Real-time thermal prediction** for clinical hyperthermia planning
- Application of SA-gPINN to problems with **sharp internal solution layers** driven by smooth sources (e.g., Burgers' equation, Allen–Cahn phase-field problems)

---

## 📚 Conclusion

This project demonstrates that **Physics-Informed Neural Networks provide an accurate and robust alternative to classical numerical methods** for solving the Dual-Phase-Lag bio-heat transfer equation. A systematic four-variant comparison — with cross-validation against Crank–Nicolson and Newmark-β agreeing to machine precision — establishes **SA-PINN as the strongest variant on sharp Gaussian sources** and validates the **proposed SA-gPINN** as the preferred choice for smooth-source problems, with a mechanistic explanation for the transition.

The work is currently being extended by a PhD mentor for **journal publication**.

---

## 📖 Citation

If you use this work, please cite:

```
Pulkit Garg, "Physics-Informed Neural Networks for Dual-Phase-Lag Bioheat Transfer:
A Comparative Study of PINN, SA-PINN, gPINN, and SA-gPINN,"
IIT (BHU) Varanasi, Supervised by Dr. Santwana Mukhopadhyay, 2025.
```

---

## 📫 Contact

**Pulkit Garg** — Department of Mathematical Sciences, IIT (BHU) Varanasi
- Email: pulkit.garg3005@gmail.com
- LinkedIn: [linkedin.com/in/pulkitgarg30](https://www.linkedin.com/in/pulkitgarg30/)
- GitHub: [github.com/Pulkitgarg30](https://github.com/Pulkitgarg30)
