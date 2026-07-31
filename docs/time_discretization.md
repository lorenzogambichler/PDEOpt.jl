# Radau IIA collocation and the resulting NLP

Time discretization and NLP structure for the full-space (direct transcription /
simultaneous) methanation OCP. The spatial FVM is taken as given: it delivers the
semi-discrete system in §1. Implementation:
[`src/timestepping/collocation.jl`](../src/timestepping/collocation.jl),
[`src/optimization/adnlp_ipopt.jl`](../src/optimization/adnlp_ipopt.jl).

---

## 1. Continuous problem

Spatial discretization leaves the quasi-linear implicit ODE system

$$
M(y)\,\dot y \;=\; f(y,t,u) \;:=\; -K(y)\,y + r(y) + b(t,u),
\qquad y(0)=y_0,\qquad y(t)\in\mathbb R^{n},
$$

with $M(y)$ diagonal and positive, $K(y)$ the transport operator, $r(y)$ the reaction
source, and $b$ affine in the scalar control $u=T_w$. For the current grid
$n = n_c\,n_f = 100\cdot 7 = 700$.

The optimal control problem is

$$
\begin{aligned}
\min_{u(\cdot)}\quad
& \frac{1}{t_f}\int_0^{t_f}\frac{F\big(y(t)\big)}{F_{\text{in}}}\,\mathrm dt
\;+\;\gamma\,\mathcal R[u] \\[2pt]
\text{s.t.}\quad
& M(y)\dot y = f(y,t,u), \qquad t\in(0,t_f], \\
& y(0) = y_0, \\
& y_{\text{lb}} \le y(t) \le y_{\text{ub}},\qquad
  u_{\text{lb}} \le u(t) \le u_{\text{ub}},\qquad t\in[0,t_f],
\end{aligned}
$$

where $F(y)=\sum_j w_j\,y_{d_j}$ is the discrete outlet CO₂ flow (`co2_w`, `co2_dofs`)
and $F_{\text{in}}$ the fixed inlet flow. Since the conversion is
$X_{\mathrm{CO_2}}(y) = 1 - F(y)/F_{\text{in}}$, minimizing the mean outlet fraction is
equivalent to maximizing the mean conversion — the Lagrange objective of Bremer et al.

---

## 2. Radau IIA collocation

### 2.1 Method

Partition $[0,t_f]$ into $N_e$ elements $I_k=[t_{k-1},t_k]$, $t_k = k\Delta t$,
$\Delta t = t_f/N_e$. On $I_k$ approximate $y$ by a polynomial
$p_k \in \mathbb P_s(I_k;\mathbb R^n)$ subject to $p_k(t_{k-1}) = y_{k-1}$, and collocate
the ODE at the $s$ interior nodes

$$
\tau_k^i \;=\; t_{k-1} + c_i\,\Delta t,\qquad i = 1,\dots,s .
$$

Write the stage values $Y_k^i := p_k(\tau_k^i)$. The Radau IIA abscissae are the roots of
the right Radau polynomial; for $s=3$

$$
c \;=\; \Big(\tfrac{4-\sqrt6}{10},\;\tfrac{4+\sqrt6}{10},\;1\Big)^{\!\top}.
$$

The Butcher matrix of a collocation method is $a_{ij}=\int_0^{c_i}\ell_j(\sigma)\,\mathrm d\sigma$
with $\ell_j$ the Lagrange basis on $\{c_1,\dots,c_s\}$; equivalently, and as constructed
in `RadauIIA`, $A$ is the unique solution of the exactness conditions

$$
\sum_{j=1}^{s} a_{ij}\,c_j^{\,m-1} \;=\; \frac{c_i^{\,m}}{m},
\qquad m = 1,\dots,s
\qquad\Longleftrightarrow\qquad
AV = W,\quad V_{jm}=c_j^{m-1},\ \ W_{im}=c_i^m/m .
$$

$$
A=\begin{pmatrix}
\frac{88-7\sqrt6}{360} & \frac{296-169\sqrt6}{1800} & \frac{-2+3\sqrt6}{225}\\[4pt]
\frac{296+169\sqrt6}{1800} & \frac{88+7\sqrt6}{360} & \frac{-2-3\sqrt6}{225}\\[4pt]
\frac{16-\sqrt6}{36} & \frac{16+\sqrt6}{36} & \frac19
\end{pmatrix},
\qquad b^\top = A_{s,:} .
$$

Properties used below: classical order $2s-1=5$, stage order $s=3$, $R(-\infty)=0$
(**L-stable**), and $c_s = 1$ with $a_{sj}=b_j$ (**stiffly accurate**).

### 2.2 Differentiation-matrix form

Since $p_k$ has degree $s$ and $\dot p_k$ degree $s-1$, the stage derivatives are
determined by their values at the $s$ nodes, giving $Y - \mathbb 1\,y_{k-1} = A\,\dot Y\,\Delta t$.
$A$ is invertible for Radau IIA, so with

$$
D := A^{-1},\qquad d_{0i} := \sum_{j=1}^{s} d_{ij},
$$

the scaled stage derivatives are

$$
\Delta t\;\dot p_k(\tau_k^i) \;=\; \sum_{j=1}^{s} d_{ij}\big(Y_k^j - y_{k-1}\big).
$$

Imposing the ODE at each collocation node gives the **stage equations**

$$
\boxed{\;
M\big(Y_k^i\big)\sum_{j=1}^{s} d_{ij}\big(Y_k^j - y_{k-1}\big)
\;+\;\Delta t\Big[K\big(Y_k^i\big)Y_k^i - r\big(Y_k^i\big) - b\big(\tau_k^i,u_k\big)\Big] \;=\; 0 \;}
$$

for $k=1,\dots,N_e$, $i=1,\dots,s$. Stiff accuracy makes the element endpoint a stage
rather than a new unknown:

$$
y_k \;:=\; Y_k^{\,s},\qquad y_0 \ \text{given}.
$$

### 2.3 Why this form and not the Butcher form

For constant $M$ the two are algebraically identical: left-multiplying the Butcher form
$M(Y^i-y_{k-1}) = \Delta t\sum_j a_{ij}f(Y^j)$ by $D=A^{-1}$ returns the boxed equation.
Two reasons to prefer the $D$ form here:

1. **State-dependent mass matrix.** $M=M(y)$ is genuinely state dependent
   (`assemble_mass!` is called from `StateAssembly`). The $D$ form is the direct
   statement of $M(y)\dot y=f(y,t)$ at each collocation node, so $M$ and $f$ are
   evaluated at the same argument. The Butcher form forces an arbitrary choice of where
   to evaluate $M$.
2. **Sparsity.** In the $D$ form, $Y_k^j$ for $j\ne i$ enters stage $i$ *only* through
   $M(Y_k^i)\,d_{ij}$ — and $M$ is diagonal. In the Butcher form $Y_k^j$ enters through
   $f(Y_k^j)$, so all $s^2$ stage blocks carry the full assembly stencil. See §5.

---

## 3. Discrete variables and constraints

### 3.1 Variable vector

Because $y_k = Y_k^s$, the trajectory is stored as $y_0$ followed by $s$ stage vectors
per element, with no duplicated endpoints:

$$
z \;=\; \big(\,\underbrace{y_0}_{n},\;
\underbrace{Y_1^1,\dots,Y_1^s}_{sn},\;\dots,\;
\underbrace{Y_{N_e}^1,\dots,Y_{N_e}^s}_{sn},\;
\underbrace{u_1,\dots,u_{N_e}}_{N_e}\,\big)
\;\in\;\mathbb R^{\,n(sN_e+1)+N_e}.
$$

Viewing the state part as an $n\times(sN_e+1)$ matrix, $Y_k^i$ is column $1+(k-1)s+i$ and
the left endpoint $y_{k-1}$ of element $k$ is column $1+(k-1)s$ (column 1 for $k=1$).

$$
n_{\text{var}} = n\,(sN_e+1)+N_e,
\qquad
n_{\text{con}} = n\,s\,N_e .
$$

### 3.2 Scaling

States and control are stored scaled, $y = y_{\text{off}} + \Sigma_y\hat y$ and
$u = u_{\text{off}} + \sigma_u\hat u$ with $\Sigma_y=\operatorname{diag}(s_y)$:

$$
\rho_\alpha = \big(c_{\text{tot,in}}M_\alpha\big)\hat\rho_\alpha,\qquad
T = T_{\min} + (T_{\max}-T_{\min})\hat T,\qquad
T_w = T_{w,\min} + (T_{w,\max}-T_{w,\min})\hat u .
$$

The offset cancels in the $D$ combination, because $\sum_j d_{ij} - d_{0i} = 0$:

$$
\sum_j d_{ij}\big(Y_k^j - y_{k-1}\big)
\;=\;\Sigma_y\Big[\sum_j d_{ij}\hat Y_k^j - d_{0i}\,\hat y_{k-1}\Big],
$$

so it may be formed on the scaled variables and rescaled by $s_y$. Constraint rows are
scaled by $\Sigma_c=\operatorname{diag}(s_c)$, $s_c$ being the reciprocal of the cell
capacity ($\varepsilon V s_y$ for species, $(\rho c_p)_{\text{ref}} V \Delta T$ for
temperature), so all rows carry the units of a scaled state.

### 3.3 Equality constraints

$$
c_k^i(z) \;=\; \Sigma_c\Big[\,
M(Y_k^i)\textstyle\sum_j d_{ij}\big(Y_k^j-y_{k-1}\big)
+\Delta t\big(K(Y_k^i)Y_k^i - r(Y_k^i) - b(\tau_k^i,u_k)\big)\Big] \;=\; 0 .
$$

These are the only constraints; there are $n s N_e$ of them, and the continuous ODE has
been fully absorbed, leaving a purely algebraic NLP.

### 3.4 Bounds

| variable | bound |
|---|---|
| $y_0$ | $l = u = \hat y_0$ (initial condition, pinned by equal bounds) |
| $\hat\rho_\alpha$ at every $Y_k^i$ | $\ge$ `x_floor` (`x_floor_H2` for H₂), no upper bound |
| $\hat T$ at every $Y_k^i$ | $T_{\min}\le T\le T_{\max}$ |
| $\hat u_k$ | $T_{w,\min}\le T_w\le T_{w,\max}$ |

Imposing the temperature box on **every stage variable** enforces the path constraint at
all $sN_e$ collocation points rather than only at element endpoints — relevant because
$T_{\max}=750\,\mathrm K$ is a safety bound on a reactor with a mobile hot spot, and the
objective actively pushes against it.

Ipopt eliminates the $n$ pinned $y_0$ entries as fixed variables, so it reports
$n_{\text{var}}-n$ free variables.

---

## 4. Objective discretization

### 4.1 Lagrange term

The Radau weights $b_i = a_{si}$ are the quadrature weights of the $s$-point Radau rule,
exact for polynomials of degree $\le 2s-2$ (order $2s-1$), with $\sum_i b_i = 1$. Hence

$$
\int_{I_k}\phi(t)\,\mathrm dt \;\approx\; \Delta t\sum_{i=1}^{s} b_i\,\phi(\tau_k^i),
$$

and since $\Delta t/t_f = 1/N_e$,

$$
J_{\text{conv}}
\;=\;\frac{1}{N_e}\sum_{k=1}^{N_e}\sum_{i=1}^{s} b_i\,\frac{F(Y_k^i)}{F_{\text{in}}}
\;\approx\;\frac{1}{t_f}\int_0^{t_f}\frac{F(y)}{F_{\text{in}}}\,\mathrm dt
\;=\; 1-\overline{X}_{\mathrm{CO_2}} .
$$

Minimizing $J_{\text{conv}}$ maximizes the mean CO₂ conversion. Note this is a genuine
order-$2s-1$ quadrature, replacing the trapezoidal rule over time nodes used with the
previous midpoint scheme.

### 4.2 Control parametrization and regularization

The control is **piecewise constant on each element**,

$$
u(t) = u_k \quad\text{for } t\in[t_{k-1},t_k),\qquad k=1,\dots,N_e,
$$

so it appears in the stage equations only as $b(\tau_k^i,u_k)$. The regularizer penalizes
element-to-element jumps:

$$
J_{\text{reg}} \;=\; N_e\sum_{k=1}^{N_e-1}\big(\hat u_{k+1}-\hat u_k\big)^2,
\qquad
J \;=\; J_{\text{conv}} + \gamma\,J_{\text{reg}} .
$$

The factor $N_e$ makes $\gamma$ mesh-independent: for a piecewise-constant $\hat u$
approximating a smooth control, $\hat u_{k+1}-\hat u_k \approx \dot{\hat u}\,\Delta t$, so
$N_e\sum(\Delta\hat u)^2 \approx \frac{t_f}{\Delta t}\sum \dot{\hat u}^2\Delta t^2
= t_f\!\int_0^{t_f}\dot{\hat u}^2\,\mathrm dt$. (Bremer et al. use the unnormalized
$\mathbf w^\top\sum(\Delta u)^2$, which is mesh-dependent — harmless at a fixed mesh.)

> **Why not one control per time node.** The previous formulation carried one $u$ per time
> node and fed the midpoint average $\bar u_k=\tfrac12(u_k+u_{k+1})$ to the dynamics. That
> averaging map $\mathcal A:\mathbb R^{N}\to\mathbb R^{N-1}$ has rank $N-1$ and kernel
> $\operatorname{span}\{(1,-1,1,-1,\dots)\}$: the alternating sawtooth is invisible to the
> constraints, and $J_{\text{conv}}$ depends only on the state. At $\gamma=0$ the solution
> was therefore non-unique along that direction with a singular reduced Hessian, so
> $\gamma$ was doing structural work. One control per element removes the kernel
> (the map is the identity) and reduces the control dimension from $sN_e+1$ to $N_e$.

---

## 5. NLP structure

### 5.1 Block sparsity

Stage residuals of element $k$ depend on $Y_k^{1..s}$, on $y_{k-1}=Y_{k-1}^{s}$, and on
$u_k$. Nothing couples across elements, so $\partial c/\partial z$ is block bidiagonal in
the element index with $sn\times sn$ diagonal blocks. Within an element, writing
$S^\star$ for the pattern of the single-argument assembly Jacobian
$\partial\big[M(y)v+\Delta t(K(y)y-r(y))\big]/\partial y$:

$$
\frac{\partial c_k^i}{\partial Y_k^j} =
\begin{cases}
\text{pattern } S^\star & j = i,\\[2pt]
d_{ij}\,M(Y_k^i) \ \ (\textbf{diagonal}) & j \ne i,
\end{cases}
\qquad
\frac{\partial c_k^i}{\partial y_{k-1}} = -\,d_{0i}\,M(Y_k^i)\ \ (\textbf{diagonal}),
$$

$$
\frac{\partial c_k^i}{\partial u_k} = -\Delta t\,\Sigma_c\,\frac{\partial b}{\partial u}
= -\Delta t\,\Sigma_c f_{\text{wall}},\qquad
\operatorname{nnz} = |\operatorname{supp} f_{\text{wall}}| = n_z .
$$

Only the $s$ diagonal stage blocks carry the assembly stencil; the $s(s-1)$ off-diagonal
stage blocks and the endpoint block are diagonal. This is the payoff of the $D$ form
(§2.3): the Butcher form would put $S^\star$ in all $s^2$ blocks.

Per element,

$$
\operatorname{nnz} \;=\; s\,|S^\star| \;+\; s(s-1)\,n \;+\; s\,n \;+\; s\,n_z .
$$

### 5.2 Measured structure

Measured on element $k=6$ of a $N_e=12$, $s=3$, $n=700$ transcription
(`jac` of the assembled `ADNLPModel`):

| block | nnz | structure |
|---|---|---|
| $\partial c^i/\partial Y^i$ | 14 746 | assembly stencil $S^\star$ (21.1 nnz/row) |
| $\partial c^i/\partial Y^j,\ j\ne i$ | 700 | diagonal |
| $\partial c^i/\partial y_{k-1}$ | 700 | diagonal |
| $\partial c^i/\partial u_k$ | 25 | wall cells only |

Predicted $3(14746)+6(700)+3(700)+3(25) = 50\,613$; measured 50 613 exactly.

> **Structural zeros are not free.** `M` and `Jr` are allocated as `copy(K)` so that
> Crank–Nicolson can combine their `nzval` arrays elementwise, but `assemble_mass!` writes
> only the diagonal. A `mul!` with `M` walks *all stored* entries, so the sparsity tracer
> recorded a dependency for every structural zero; likewise
> `@. b += f_wall * u` is a dense broadcast over a vector with $n_z$ nonzeros. Replacing
> these by a diagonal product over the precomputed `mdiag` and a scatter over
> `wall_dofs` is numerically exact but cuts the Jacobian from 111 984 to 50 613 nnz per
> element — a factor **2.21**. AD sparsity follows the operations executed, not the values.

### 5.3 Sizes

With $n=700$, $s=3$, $\Delta t = 25\ \mathrm s$:

| $t_f$ [s] | $N_e$ | $n_{\text{var}}$ | $n_{\text{con}}$ | Jacobian nnz |
|---|---|---|---|---|
| 600 | 24 | 51 124 | 50 400 | 1 214 712 |
| 1000 | 40 | 84 740 | 84 000 | 2 024 520 (measured) |

For comparison, implicit midpoint at $\Delta t = 1\ \mathrm s$ has $2|S^\star| + 2n_z
= 29\,542$ nnz per step (same measured stencil), giving at $t_f=600$ about
$4.2\times10^5$ variables and $1.8\times10^7$ nonzeros — **8× the variables and 15× the
nonzeros of the Radau transcription, at order 2 instead of 5 and without L-stability.**

---

## 6. Verification

All figures below are measured, not estimated.

**Tableau** (`RadauIIA(s)`, $s=1,2,3$). $A$ reproduces the tabulated Radau IIA matrices;
$\sum_j a_{ij}=c_i$; $b = A_{s,:}$; the quadrature satisfies
$\sum_i b_i c_i^{m-1}=1/m$ for $m=1,\dots,2s-1$; and
$R(-\infty)=1-b^\top A^{-1}\mathbb 1 = 0$ to machine precision.

**Order and damping** on $\dot y=\lambda y$ using the same $D$-form residual, i.e.
$(D-\Delta t\lambda I)\,Y = d_0\,y_{k-1}$:

| $s$ | expected order | observed |
|---|---|---|
| 1 | 1 | 1.54, 1.20, 1.09, 1.04 |
| 2 | 3 | 3.11, 3.05, 3.03, 3.01 |
| 3 | 5 | 5.07, 5.03, 5.02, 5.01 |

Single-step amplification, $s=3$: $|R| = 2.5\!\times\!10^{-2},\,3.0\!\times\!10^{-4},\,
3.0\!\times\!10^{-6},\,3.0\!\times\!10^{-8}$ at $\Delta t\lambda = -10^2,-10^4,-10^6,-10^8$,
i.e. $|R|\sim 3/|\Delta t\lambda|\to 0$. Implicit midpoint on the same points gives
$0.96,\,0.9996,\,1.000,\,1.000$ — the stiff modes are reflected, not damped.

**Transcription against a reference integrator.** With $u$ fixed and $y_0$ pinned, Newton
on the stage variables converges quadratically,
$\|c\|_\infty: 2.71\times10^{-2}\to2.14\times10^{-7}\to2.45\times10^{-12}$,
confirming the residual and its AD Jacobian are consistent. Comparing the resulting
element endpoints against Crank–Nicolson at $\Delta t = 0.25\ \mathrm s$ over
$t_f = 300\ \mathrm s$, with Radau at $\Delta t = 25\ \mathrm s$ (a 100× coarser mesh):

| $t$ [s] | 25 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|
| rel. state error | 8.1e-4 | 4.2e-5 | 7.2e-7 | 1.6e-7 | 9.4e-8 |
| hot-spot $\Delta T$ [K] | 3.1e-1 | 2.1e-2 | 2.7e-4 | 4.7e-5 | 5.8e-5 |

The largest error sits on the first element, where the start-up transient is steepest;
once the profile smooths the agreement is 6–7 digits.

---

## 7. Relation to Bremer et al.

The paper solves the same class of problem with the same machinery — Radau collocation,
simultaneous transcription, IPOPT with MA97 — and the mesh here matches theirs:
**3 collocation points, $\Delta t = 25\ \mathrm s$** (they use 40 elements over 1000 s).
Their reported $n = 16\cdot5\cdot7 = 560$ against $n = 700$ here.

Differences worth recording:

- **Control.** They use two controls, $T_{\text{cool,in}}$ and $T_{\text{cool,out}}$, with
  $T_{\text{cool}}(t,z)$ linear in $z$ between them; here $T_w$ is uniform in $z$, a single
  scalar control. Adding the axial profile means $b(t,u)=b_0(t)+B u$ with
  $B\in\mathbb R^{n\times2}$ and doubles the control columns — no structural change.
- **Control mesh.** They state the control is piecewise constant *between collocation
  points* ($s N_e$ values per input); this implementation uses one value per element
  ($N_e$). Per-element is the more common choice and avoids control degrees of freedom on
  the non-uniform sub-intervals induced by $c$.
- **Grid.** They grade the axial FVM logarithmically (59 mm → 872 mm) to resolve the inlet
  gradient; `setup_problem` here is uniform. Their mesh study puts $16\times5$ within 2 %
  of the hot-spot temperature at $40\times25$.
- **Objective.** Same Lagrange conversion integral plus a squared control-jump penalty;
  their $w = 10^{-3}$ against $\gamma = 10^{-2}$ here, on differently normalized terms, so
  the values are not directly comparable.

Their qualitative result — the optimizer first steps $T_{\text{cool}}$ to the upper bound
to ignite, then lowers it ~450 s *before* the hot spot peaks, riding $T_{\max}$ thereafter
— is the behaviour to check this transcription against.
