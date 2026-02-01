# Dual-Phase-Lag Bio-Heat Transfer Modeling using Physics-Informed Neural Networks (PINN)

## 📌 Project Overview
This project focuses on the numerical modeling of bio-heat transfer in living biological tissue using the **Dual-Phase-Lag (DPL) bio-heat transfer model**. The DPL model extends the classical Pennes bio-heat equation by incorporating **finite thermal wave propagation** and **micro-scale lag effects**, making it more suitable for fast transient thermal processes such as **hyperthermia cancer treatment**.

The study implements and compares three different solution approaches:
- **Physics-Informed Neural Networks (PINN)**
- **Crank–Nicolson (CN) finite difference scheme**
- **Newmark-β time integration method**

The objective is to validate the accuracy, stability, and physical consistency of the PINN solution by benchmarking it against well-established classical numerical methods.

---

## 🎯 Motivation
Thermal therapies such as hyperthermia require precise prediction and control of temperature distribution inside biological tissue. Classical models based on Fourier’s law assume infinite speed of heat propagation, which is physically unrealistic for:
- Rapid heating processes
- Micro-scale tissue interactions
- Short-time thermal responses

The **Dual-Phase-Lag (DPL) model** resolves these limitations by introducing:
- A **phase lag of heat flux (τq)**  
- A **phase lag of temperature gradient (τT)**  

These additions allow the model to capture wave-like thermal behavior and provide more accurate predictions for biomedical applications.

---

## 📐 Mathematical Formulation

### Dual-Phase-Lag Bio-Heat Equation (Dimensionless Form)
The governing equation solved in this project is the dimensionless DPL bio-heat transfer equation:

\[
F_{oq}\frac{\partial^2 \theta}{\partial Fo^2}
+ (1 + F_{oq}P_f^2)\frac{\partial \theta}{\partial Fo}
+ P_f^2 \theta
=
\frac{\partial^2 \theta}{\partial x^2}
+ Fo_T \frac{\partial^3 \theta}{\partial x^2 \partial Fo}
+ P_f^2 \theta_b + P_m + P_r
\]

Where:
- \( \theta \) is the dimensionless temperature
- \( Fo \) is the Fourier number (dimensionless time)
- \( x \) is the dimensionless spatial coordinate
- \( F_{oq} \) represents the heat-flux phase lag
- \( Fo_T \) represents the temperature-gradient phase lag
- \( P_f \) is the blood perfusion parameter

A **Gaussian heat source** is used to model localized tumor heating.

---

## 🧪 Initial and Boundary Conditions
- **Initial conditions**:
  - \( \theta(x,0) = 0 \)
  - \( \frac{\partial \theta}{\partial Fo}(x,0) = 0 \)

- **Boundary conditions**:
  - Symmetry (Neumann) condition at the tissue center:  
    \( \frac{\partial \theta}{\partial x} = 0 \) at \( x = 0 \)
  - Robin (third-kind) condition at the outer boundary:  
    \( A\frac{\partial \theta}{\partial x} + B\theta = g(Fo) \) at \( x = 1 \)

---

## 🤖 Physics-Informed Neural Network (PINN) Approach

In the PINN framework, a deep neural network is trained to approximate the temperature field while being constrained by the governing physics.

### Key Characteristics
- Mesh-free solution method
- Governing PDE enforced through the loss function
- Boundary conditions enforced through residual minimization
- Initial conditions satisfied **exactly** using a trial solution

### Trial Solution
To automatically satisfy the initial conditions, the neural network output is defined as:
\[
\theta_{NN}(x, Fo) = Fo^2 \cdot N(x, Fo)
\]

### Loss Function
The total loss function consists of:
- PDE residual loss
- Boundary condition loss (symmetry and Robin conditions)

---

## 🧮 Classical Numerical Methods

### Crank–Nicolson Method
- Second-order accurate in time
- Unconditionally stable
- Implicit finite difference scheme
- Requires solving a coupled linear system at each time step

### Newmark-β Method
- Designed for second-order hyperbolic systems
- Parameters used: \( \beta = \frac{1}{4}, \gamma = \frac{1}{2} \)
- Second-order accurate
- Excellent energy conservation
- Unconditionally stable

These methods serve as reference solutions for validating the PINN results.

---

## 📊 Results and Validation

### Qualitative Results
- All methods capture localized heating at the tumor location (\( x = 0.5 \))
- Temperature profiles exhibit a bell-shaped spatial distribution
- Smooth and physically realistic temporal evolution of temperature

### Quantitative Error Analysis

| Comparison | Relative L₂ Error | Relative L∞ Error |
|----------|------------------|------------------|
| Crank–Nicolson vs PINN | ~1.1 × 10⁻² | ~1.6 × 10⁻² |
| Newmark-β vs PINN | ~1.1 × 10⁻² | ~1.6 × 10⁻² |
| CN vs Newmark-β | ~1 × 10⁻¹¹ | ~1 × 10⁻¹¹ |

- CN and Newmark-β solutions are numerically identical
- PINN achieves agreement within approximately **1–2% error**

---

## ✅ Boundary Condition Verification
- Symmetry and Robin boundary conditions are satisfied within an error tolerance of **O(10⁻³)**
- Residual analysis confirms strong physical consistency
- No explicit penalty tuning is required for boundary enforcement

---

## 🛠 Technologies Used
- Python
- Physics-Informed Neural Networks (PINN)
- TensorFlow / PyTorch
- NumPy, SciPy
- Matplotlib
- Finite Difference Methods

---



## 🚀 Key Contributions
- Demonstrated accurate PINN-based solution of the DPL bio-heat equation
- Validated PINN against two classical, unconditionally stable numerical schemes
- Showed that PINNs can reliably solve hyperbolic-parabolic PDEs
- Provided a mesh-free and continuous solution framework

---

## 🔮 Future Scope
- Extension to two- and three-dimensional tissue geometries
- Inverse PINN for parameter estimation
- Patient-specific hyperthermia modeling
- Real-time thermal prediction for clinical applications

---

## 📚 Conclusion
This project demonstrates that **Physics-Informed Neural Networks** provide an accurate and robust alternative to classical numerical methods for solving the **Dual-Phase-Lag bio-heat transfer equation**. The strong agreement with Crank–Nicolson and Newmark-β solutions confirms the reliability of the PINN approach while offering additional flexibility for complex geometries and future biomedical applications.
