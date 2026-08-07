# Exact Lagrangian Hessian by element-level assembly

Reference for the second-derivative implementation of the methanation NLP. The NLP itself
(variables, constraints, scaling, objective) is defined in
[`time_discretization.md`](time_discretization.md); this document assumes §3 and §4 of it
and only adds what is needed for $\nabla^2 L$.

Implementation:
[`src/optimization/methanation_hessian.jl`](../src/optimization/methanation_hessian.jl),
kernels in [`src/models/methanation.jl`](../src/models/methanation.jl) and
[`src/assembly/assemble_fvm.jl`](../src/assembly/assemble_fvm.jl).

---

## 1. Notation

Fixed throughout. Code identifiers are given in `monospace`.

**Mesh and fields.**

| symbol | meaning | code |
|---|---|---|
| $n_c$ | number of cells | `ncells(grid)` |
| $n_f$ | fields per cell, $=7$: $(\rho_{\mathrm{CH_4}},\dots,\rho_{\mathrm{N_2}},T)$ | `length(dm.fields)` |
| $a$ | field index, $1\le a\le n_f$; $a=n_f$ **is** temperature | — |
| $d(c,a)=(c-1)n_f+a$ | spatial dof index | `dof(dm,c,a)` |
| $\mathcal C_c=\{d(c,1),\dots,d(c,n_f)\}$ | the $n_f$ dofs of cell $c$ (contiguous) | `celldof(dm,c)` |
| $n=n_c n_f$ | spatial dofs | `ocp.n` |
| $\mathcal F$ | interior faces, each $e\in\mathcal F$ with owner $o(e)$, neighbour $q(e)$, axis $\in\{1,2\}$ | `sa.dirs` |

**Transcription.** $s$ stages, $N_e$ elements, $\Delta t$, tableau $D=(d_{ij})$,
$d_{0i}=\sum_j d_{ij}$, as in `time_discretization.md` §2.2.

| symbol | meaning | code |
|---|---|---|
| $\mathrm{col}(k,i)=1+(k-1)s+i$ | column of $\hat Y_k^i$ in the state matrix | `stagecol(ocp,k,i)` |
| $\mathrm{col}_0(k)=1+(k-1)s$ | column of $\hat y_{k-1}$ | `leftcol(ocp,k)` |
| $g(d,\mathrm{col})=(\mathrm{col}-1)n+d$ | **variable index** of $\hat Z_{d,\mathrm{col}}$ in $z$ | `gidx(ocp,d,col)` |
| $g_u(k)=n(sN_e+1)+k$ | variable index of $\hat u_k$ | — |
| $R(k,i,d)=\big((k-1)s+i-1\big)n+d$ | **constraint index** of row $d$ of stage $(k,i)$ | — |
| $n_{\mathrm{var}}=n(sN_e+1)+N_e$, $n_{\mathrm{con}}=nsN_e$ | | `nvars`, `ncons` |

**Scaling.** $s_y\in\mathbb R^n$ (`ocp.sy`), $y_{\mathrm{off}}\in\mathbb R^n$
(`ocp.y_off`, nonzero only on $T$ dofs), $s_c\in\mathbb R^n$ (`ocp.sc`),
$\Sigma_y=\operatorname{diag}(s_y)$, $\Sigma_c=\operatorname{diag}(s_c)$. Hats denote
scaled quantities; $z$ stores hatted variables throughout.

**Stage quantities.** For element $k$, stage $i$:

$$
y^{k,i} \;=\; y_{\mathrm{off}}+\Sigma_y\,\hat Y_k^i \;\in\;\mathbb R^n,
\qquad
w^{k,i} \;=\; \Sigma_y\Big[\sum_{j=1}^{s} d_{ij}\hat Y_k^j - d_{0i}\,\hat y_{k-1}\Big]
\;\in\;\mathbb R^n .
$$

$y^{k,i}$ is the **physical** state at the collocation node; $w^{k,i}$ is the (physical)
scaled stage derivative $\Delta t\,\dot p_k(\tau_k^i)$. Restrictions to a cell carry a
subscript: $y^{k,i}_c := (y^{k,i}_d)_{d\in\mathcal C_c}\in\mathbb R^{n_f}$.

**Multipliers.** $\lambda\in\mathbb R^{n_{\mathrm{con}}}$, objective weight $\sigma$, and
the *row-scaled multiplier*, which absorbs $\Sigma_c$ once and for all:

$$
\boxed{\;\nu^{k,i}_d \;:=\; \lambda_{R(k,i,d)}\;(s_c)_d\;}
\qquad
\nu^{k,i}\in\mathbb R^{n}.
$$

Every formula below uses $\nu$, never $\lambda$ and $s_c$ separately (`ν` in
`hess_values!`).

---

## 2. What is being computed

Ipopt, through `NLPModels`, requires the **lower triangle** of

$$
\nabla^2 L(z,\lambda,\sigma)
\;=\;\sigma\,\nabla^2 f(z)\;+\;\sum_{R=1}^{n_{\mathrm{con}}}\lambda_R\,\nabla^2 c_R(z)
\;\in\;\mathbb R^{n_{\mathrm{var}}\times n_{\mathrm{var}}}
$$

in coordinate format, as two calls:

- `hess_structure!(nlp, rows, cols)` — once, fixed for the lifetime of the model;
- `hess_coord!(nlp, x, y, vals; obj_weight)` — once per iteration, `vals[t]` being the
  entry at `(rows[t], cols[t])`.

The pattern `(rows, cols)` is *declared*, not discovered: it must contain every entry that
can ever be nonzero. Declaring extra entries is correct but costs fill-in in the MA97
factorization; omitting a true entry is silently wrong.

---

## 3. Kernel decomposition of the constraint

### 3.1 Operator split

The assembled transport operator separates by provenance:

$$
K(y) \;=\; \underbrace{K^{\mathrm{f}}(y)}_{\text{interior faces}}
\;+\;\underbrace{K^{\mathrm{bc}}}_{\text{constant}}
\;+\;\underbrace{K^{\mathrm{e}}(y)}_{\text{inlet/outlet energy}} ,
$$

- $K^{\mathrm{f}}(y)$ from `assemble_transport!` — one contribution per interior face;
- $K^{\mathrm{bc}}$ from `assemble_inlet_outlet!` (**species only**, `fields=SPECIES`) plus
  `assemble_wall!`. State-**independent**, built once in `setup_problem`, stored as
  `sa.K_bc`;
- $K^{\mathrm{e}}(y)$ from `assemble_energy_advection!` — the $T$ row of inlet/outlet
  boundary cells only.

The forcing splits the same way:

$$
b^{k,i} \;=\; f_{\mathrm{in}}(y^{k,i})\odot m(\tau_k^i)\;+\;f_{\mathrm{wall}}\,u_k ,
$$

where $m(t)_d = \texttt{inlet\_mod}(m,\text{field}(d),t)$, $f_{\mathrm{wall}}$ is constant,
and $f_{\mathrm{in}}$ is **constant on species dofs but state-dependent on the $T$ dofs of
inlet cells** — also written by `assemble_energy_advection!`. This asymmetry is easy to
miss: `b` reads like a pure forcing term but carries curvature.

### 3.2 The four kernels

All state dependence factors through four pure, allocation-free functions. Nothing in the
model couples more than two cells.

| kernel | signature | code |
|---|---|---|
| $P_c=\mathcal P(y_c)$ | $\mathbb R^{n_f}\to(\rho,\rho_M,p,x,c,D^{\mathrm{eff}},\rho c_p,\rho c_p^{\mathrm{flow}},\lambda^{\mathrm{eff}})$ | `props_cell` |
| $\mathfrak r_c=\mathcal R(y_c)$ | $\mathbb R^{n_f}\to\mathbb R^{n_f}$ volumetric source | `reaction_cell` |
| $F_e=\mathcal F_e(y_{o},y_{q})$ | $\mathbb R^{2n_f}\to\mathbb R^{n_f}$ face flux | `face_flux` |
| $(\kappa_c,\phi_c)=\mathcal B_c(y_c)$ | $\mathbb R^{n_f}\to\mathbb R^2$, ($K^{\mathrm e}$ diagonal, $f_{\mathrm{in}}$ value) | `bnd_energy_kernel` |

**$\mathcal P$ and $\mathcal R$ are the only nonlinear cell maps, $\mathcal F_e$ the only
nonlinear face map.** Everything downstream is *bilinear* in $(\text{properties},y)$:

$$
M_{dd}=V_c\,\underbrace{\big[a=n_f\big]\,\rho c_p + \big[a<n_f\big]\,\varepsilon}_{\texttt{capacity\_cell}},
\qquad
K^{\mathrm f}_{oq}\propto\big(\theta_e P_{o}+(1-\theta_e)P_{q}\big),
$$

with $V_c$ the cell volume and $\theta_e$ the face interpolation weight (`fg.wf[e]`).

The face contribution is **antisymmetric**: face $e$ adds $+\Delta t F_{e,a}$ to row
$d(o,a)$ and $-\Delta t F_{e,a}$ to row $d(q,a)$.

### 3.3 Residual in kernel form

For $d=d(c,a)$,

$$
\begin{aligned}
\frac{c^{k,i}_d}{(s_c)_d}
\;=\;& V_c\,\mathrm{cap}_a(P_c)\,w^{k,i}_d
\;-\;\Delta t\,V_c\,\mathfrak r_{c,a}
&&\text{(cell)}\\
+\;& \Delta t\Big[\textstyle\sum_{e:\,o(e)=c}F_{e,a}\;-\;\sum_{e:\,q(e)=c}F_{e,a}\Big]
&&\text{(faces)}\\
+\;& \Delta t\,(K^{\mathrm{bc}}y^{k,i})_d
\;-\;\Delta t\,f^{\text{sp}}_{\mathrm{in},d}\,m_a
\;-\;\Delta t\,f_{\mathrm{wall},d}\,u_k
&&\text{(constant operators)}\\
+\;& \delta_{a,n_f}\;\Delta t\,\big[\kappa_c\,y^{k,i}_{d}-\phi_c\,m_{n_f}\big]
&&\text{(inlet/outlet energy)} .
\end{aligned}
$$

The last line is present only for the temperature row of an inlet/outlet cell.
This is implemented literally as `_residual_kernels!` in
`methanation_adnlp.jl`, which exists **only** to prove the kernels reproduce the global
sparse assembly `_residual!` (§9). It is not used in the solve.

---

## 4. Curvature audit

Which terms of §3.3 contribute to $\nabla^2 c$:

| term | in $z$ | Hessian |
|---|---|---|
| $\Delta t\,(K^{\mathrm{bc}}y)_d$ | linear | **0** |
| $\Delta t\,f_{\mathrm{in}}^{\text{species}}\odot m$ | constant | **0** |
| $\Delta t\,f_{\mathrm{wall}}u_k$ | linear | **0** |
| $V_c\,\varepsilon\,w_d$ (species mass) | linear ($\varepsilon$ const, $w$ affine) | **0** |
| $V_c\,\rho c_p(y_c)\,w_{d}$ ($T$ mass) | product of nonlinear × affine | $n_f$-block **+ cross-stage** |
| $-\Delta t V_c\,\mathfrak r_{c,a}$ | nonlinear in $y_c$ | $n_f$-block |
| $\Delta t[\kappa_c y_{d}-\phi_c m_{n_f}]$ | nonlinear in $y_c$ | $n_f$-block |
| $\Delta t\,F_{e,a}$ | nonlinear in $(y_o,y_q)$ | $2n_f$-block |

Two consequences worth stating explicitly:

1. **No control appears in any constraint Hessian.** $b$ is affine in $u_k$, so
   $\partial^2 c/\partial u\,\partial\,\cdot\;=0$ identically. All control curvature comes
   from the objective regularizer.
2. **Only the $T$ mass term couples across stages.** Every other block lives entirely
   inside the stage-$i$ column. This is the $D$-form payoff of `time_discretization.md`
   §2.3 propagating to second order.

---

## 5. Local blocks

### 5.1 Gather formulation

Let $\Phi:\mathbb R^{n_{\mathrm{loc}}}\to\mathbb R$ be a local scalar and
$G\in\{0,1\}^{n_{\mathrm{loc}}\times n_{\mathrm{var}}}$ the gather with
$G_{a,\,\mathfrak g(a)}=1$, where $\mathfrak g(a)$ is the global variable index of local
slot $a$. Then the block's contribution is $G^\top\big(\nabla^2\Phi\big)G$, i.e.

$$
\boxed{\;
\big[G^\top(\nabla^2\Phi)G\big]_{PQ}
\;=\!\!\sum_{a:\,\mathfrak g(a)=P}\ \sum_{b:\,\mathfrak g(b)=Q}\!\!\big(\nabla^2\Phi\big)_{ab}\;}
\tag{5.1}
$$

and the total is

$$
\nabla^2 L \;=\; \sigma\nabla^2 f \;+\;
\sum_{k=1}^{N_e}\sum_{i=1}^{s}\Big[
\sum_{c=1}^{n_c} G_{c}^{k,i\;\top}\big(\nabla^2\Phi^{k,i}_{c}\big)G_{c}^{k,i}
\;+\;\sum_{e\in\mathcal F} G_{e}^{k,i\;\top}\big(\nabla^2\Phi^{k,i}_{e}\big)G_{e}^{k,i}
\Big].
$$

$\mathfrak g$ **need not be injective** — see §6.

### 5.2 Cell block

$n_{\mathrm{loc}}=n_f+s+1$ (= 11 for $n_f=7,s=3$). Local slots and their global images:

| slot $a$ | variable | $\mathfrak g(a)$ |
|---|---|---|
| $1\dots n_f$ | $\hat Z_{d(c,a),\,\mathrm{col}(k,i)}$ | $g\big(d(c,a),\mathrm{col}(k,i)\big)$ |
| $n_f+j$, $j=1\dots s$ | $\hat Z_{d(c,n_f),\,\mathrm{col}(k,j)}$ | $g\big(d(c,n_f),\mathrm{col}(k,j)\big)$ |
| $n_f+s+1$ | $\hat Z_{d(c,n_f),\,\mathrm{col}_0(k)}$ | $g\big(d(c,n_f),\mathrm{col}_0(k)\big)$ |

Slots $1..n_f$ carry the cell's own state at stage $i$; slots $n_f+1..n_f+s+1$ carry the
$T$ dof across all stages, which is all the $T$ mass term needs from $w$. With
$v\in\mathbb R^{n_{\mathrm{loc}}}$, $\eta_a := (y_{\mathrm{off}})_{d(c,a)}$,
$\varsigma_a := (s_y)_{d(c,a)}$:

$$
y_c(v) = \big(\eta_a+\varsigma_a v_a\big)_{a=1}^{n_f},
\qquad
\omega(v) \;=\; \varsigma_{n_f}\Big[\sum_{j=1}^{s} d_{ij}\,v_{n_f+j}
   \;-\; d_{0i}\,v_{n_f+s+1}\Big],
$$

$$
\Phi^{k,i}_c(v)
\;=\;-\,\Delta t\,V_c\sum_{a=1}^{n_f}\nu^{k,i}_{d(c,a)}\,\mathcal R_a\big(y_c\big)
\;+\;\nu^{k,i}_{d(c,n_f)}\,V_c\,\rho c_p\big(y_c\big)\,\omega
\;+\;\Delta t\,\nu^{k,i}_{d(c,n_f)}\Big[\kappa_c\big(y_c\big)\,y_{c,n_f}-\phi_c\big(y_c\big)\,m_{n_f}\Big].
$$

The last bracket is present only when $c$ is an inlet/outlet cell, flagged by
$\beta_c\ne 0$ where $\beta_c = v_z\cdot\mathrm{side}$ (`_bnd_cells`). Note $\omega$ is
exactly $w^{k,i}_{d(c,n_f)}$ evaluated from the local slots.

Species mass terms are absent by the audit of §4; the scaling $\varsigma$ is applied
*inside* $\Phi$, so $\nabla^2\Phi$ is already the derivative with respect to the scaled
variables and no post-multiplication by $\Sigma_y$ is needed.

### 5.3 Face block

$n_{\mathrm{loc}}=2n_f$ (= 14). Slots $1..n_f$ are the owner's dofs, $n_f+1..2n_f$ the
neighbour's, all at column $\mathrm{col}(k,i)$:

$$
\Phi^{k,i}_e(v)
\;=\;\Delta t\sum_{a=1}^{n_f}\Big(\nu^{k,i}_{d(o,a)}-\nu^{k,i}_{d(q,a)}\Big)\,
\mathcal F_{e,a}\big(y_{o},y_{q}\big),
$$

with $y_{o},y_{q}$ built from $v$ exactly as in §5.2. The difference of multipliers is the
antisymmetry of §3.2 contracted into a single scalar, so one $14\times14$ Hessian serves
both rows of the face.

### 5.4 Objective

$J_{\mathrm{conv}}$ is **affine** in $\hat Z$ (the outlet CO₂ flow is a fixed linear
functional of the state; `time_discretization.md` §4.1), so all objective curvature comes
from $J_{\mathrm{reg}}=N_e\sum_{k=1}^{N_e-1}(\hat u_{k+1}-\hat u_k)^2$:

$$
\sigma\,\nabla^2 f\Big|_{g_u(k),\,g_u(l)}
=2\sigma\gamma N_e\times
\begin{cases}
[k>1]+[k<N_e], & l=k,\\
-1, & |l-k|=1,\\
0,&\text{otherwise,}
\end{cases}
$$

a constant tridiagonal $N_e\times N_e$ block, $2N_e-1$ entries in the lower triangle. It is
written directly, with no AD.

---

## 6. Aliased slots

In §5.2, slot $n_f$ and slot $n_f+i$ are **the same variable**:
$\mathfrak g(n_f)=\mathfrak g(n_f+i)=g\big(d(c,n_f),\mathrm{col}(k,i)\big)$. This is
deliberate — it lets one dense block carry both the cell's own curvature and the
cross-stage $M(y)w$ coupling — but it makes $\mathfrak g$ non-injective, and (5.1) then has
more than one term.

Write $p=n_f$, $p'=n_f+i$, $P=\mathfrak g(p)$. By (5.1),

$$
\big[G^\top(\nabla^2\Phi)G\big]_{PP}
=\Phi_{pp}+2\Phi_{pp'}+\Phi_{p'p'},
\qquad \Phi_{ab}:=\big(\nabla^2\Phi\big)_{ab}.
$$

The scatter loop visits each *unordered* local pair once ($b\le a$), so the cross term
must be doubled explicitly:

```julia
mult = (a != b && gv[a] == gv[b]) ? 2.0 : 1.0
vals[pos[t]] += mult * H[a, b]
```

For off-diagonal global pairs $P\ne Q$ no factor is needed: the single visit to $(a,b)$
already represents the symmetric pair, since Ipopt mirrors the lower triangle.

> **This term is currently invisible to testing.** $\Phi_{pp'}\propto \partial(\rho c_p)/\partial T$, and `DEFAULT_CP` are single-coefficient `Poly`s —
> constant in $T$ — so $\partial(\rho c_p)/\partial T\equiv0$ and the missing term is
> exactly zero. Omitting `mult` passes every test against the repo defaults. With a
> $T$-dependent $c_p$ (the `TODO: replace with VDI polynomial coeffs`) the relative error
> is $4.2\times10^{-9}$. The same defaults also make $\rho c_p^{\mathrm{flow}}$
> $T$-independent, damping the $T$-advection face block and the boundary energy kernel.
> **Any AD change must be validated against a $T$-dependent $c_p$, not the defaults.**

---

## 7. Structure assembly

`hess_structure` walks the blocks in a fixed order — for each $(k,i)$: cells $1..n_c$, then
faces in `sa.dirs` order — and for each block emits every lower-triangular local pair as a
packed key

$$
\mathrm{key}(a,b)=\big(\max(\mathfrak g(a),\mathfrak g(b))-1\big)n_{\mathrm{var}}
+\min(\mathfrak g(a),\mathfrak g(b))\;\in\;\mathbb Z,
$$

injective for $1\le \mathfrak g\le n_{\mathrm{var}}$ and bounded by
$n_{\mathrm{var}}^2\approx4.1\times10^9\ll2^{63}$. Sorting and `unique!` give the deduplicated
pattern; `searchsortedfirst` maps each emitted pair to its slot, stored flat in `pos`.
`hess_values!` walks the identical block order and accumulates
`vals[pos[t]] += mult*H[a,b]`.

**Invariant.** `hess_structure` and `hess_values!` must iterate blocks in the same order,
including the `sa.dirs` traversal. A reordering in one and not the other is silent.

The pattern is derived, not traced, so it is *tighter* than
`SparseConnectivityTracer`: 1 186 269 against 1 633 948 entries at the reference size, a
27 % reduction. It is still a mild over-approximation — structurally present entries that
happen to vanish numerically (e.g. the $\rho c_p$–$T$ entries under constant $c_p$) are
declared, which is the correct choice: the pattern must not depend on parameter values.

---

## 8. Cost

Reference size: $n_z=25$, $n_r=4$ ⇒ $n_c=100$, $n=700$; $s=3$, $N_e=30$ ⇒
$n_{\mathrm{var}}=63\,730$, $n_{\mathrm{con}}=63\,000$. Faces: $(n_z-1)n_r=96$ axial,
$n_z(n_r-1)=75$ radial, $|\mathcal F|=171$.

Blocks per Hessian: $sN_e\,(n_c+|\mathcal F|)=90\cdot271=24\,390$, of which $9\,000$ are
$11\times11$ and $15\,390$ are $14\times14$.

| | measured |
|---|---|
| structure build | 0.8 s, 0.04 GiB |
| `nnzh` | 1 186 269 |
| one `hess_values!` | 1.41 s |
| one residual (`ocp_cons!`) | 26.7 ms |
| ratio | **52.6× one residual** |
| peak RSS (full pipeline) | 0.87 GiB |

A dense $m$-variable `ForwardDiff.hessian` costs $O(m^2)$ kernel evaluations, so keeping
$m\le 2n_f=14$ rather than $m=n_{\mathrm{var}}$ is the entire reason this is affordable.
Forward-over-forward — normally the wrong algorithm at NLP scale — is the *best* algorithm
at $m=14$: no taping, no reverse sweep, no activity analysis.

---

## 9. Verification

`hess_values!` against a dense `ForwardDiff.hessian` of
$\sigma f(z)+\lambda^\top c(z)$ at random $z$, random $\lambda$, $\sigma=0.7$:

| configuration | $n_{\mathrm{var}}$ | rel. error |
|---|---|---|
| $s=1,N_e=1$ | 141 | 2.63e-16 |
| $s=1,N_e=2$ | 212 | 3.61e-16 |
| $s=2,N_e=2$ | 352 | 2.76e-16 |
| $s=3,N_e=2$ | 492 | 3.46e-16 |
| $s=3,N_e=2,n_r=3$ | 737 | 3.18e-16 |
| $s=3,N_e=2$, $T$-dependent $c_p$ | 492 | 2.44e-16 |

with `true nonzeros outside declared pattern: 0` in every case — the *structure* is
verified, not only the values.

Upstream, `_residual_kernels!` matches `_residual!` to $2\times10^{-15}$ relative and its
Jacobian to $1.4\times10^{-16}$ with identical sparsity (8082 = 8082 nnz). Exact agreement
is impossible: `_residual!` accumulates via a CSC mat-vec in column order, the kernel loop
face by face, and floating-point addition is not associative.

> Do not use the uniform inlet state $z_0$ as a test point. The residual there nearly
> cancels ($\|c\|=20.8$ from terms of magnitude $515$), so relative errors inflate by the
> cancellation factor and look like failures at $10^{-13}$.

Through the `KernelHessNLP` wrapper (`hess_structure!` / `hess_coord!`), against the same
dense reference: $1.24\times10^{-16}$ with $\lambda\ne0$, and exactly $0$ on the
objective-only path ($\lambda=0$), where the tridiagonal control block is the whole answer.

---

## 10. Convergence behaviour

Solve at $n_z=10$, $n_r=3$, $N_e=8$, $s=3$ ⇒ $n_{\mathrm{var}}=5258$,
`tol` $=10^{-8}$, MA97, both runs from the same $z_0$:

| | iterations | wall clock | objective |
|---|---|---|---|
| exact Hessian | 59 | 53.1 s | 0.747775422548136 |
| L-BFGS, history 75 | 55 | 22.4 s | 0.747775422548447 |

Same solution ($|\Delta J|=3.1\times10^{-13}$). **At this size the exact Hessian is 2.4×
slower in wall clock and does not reduce the iteration count.** The Ipopt timing breakdown
explains it: `LinearSystemFactorization` is 24.5 s of the 31.5 s spent inside Ipopt,
because the Hessian contributes 89 217 nonzeros to a KKT matrix that limited-memory mode
leaves nearly empty. The exact Hessian is not free curvature — it is curvature paid for in
factorization fill.

What it does buy is the asymptotic rate. The tail of the exact run:

| iter | `inf_pr` | `inf_du` |
|---|---|---|
| 56 | 1.39e-03 | 3.57e-03 |
| 57 | 1.09e-05 | 5.85e-04 |
| 58 | 2.60e-08 | 5.55e-07 |
| 59 | 1.14e-12 | 7.31e-12 |

Each residual is roughly the square of the previous — Newton's quadratic rate, which a
rank-$2m$ secant approximation cannot reproduce. Final dual infeasibility
$7.3\times10^{-12}$ against a requested $10^{-8}$; the last three digits of accuracy are
essentially free once the rate kicks in. That matters when the run is *hard* — tight
$\gamma$, an active $T_{\max}$ path constraint, or the stalling near the optimum that
motivated this work — and does not matter on a problem L-BFGS already solves comfortably.

**Choose accordingly:** limited-memory for exploratory runs and coarse meshes, the exact
Hessian when the tail stalls or high accuracy is needed.

---

## 11. Invariants

Things that must remain true; each has already been violated once during development.

1. **No closure may capture a loop-mutated accumulator.** Julia boxes it (`Core.Box`),
   the value becomes `::Any`, and allocation explodes — 45 MB per residual in the case of
   `ρM` in `props_cell`. Invisible to `code_warntype` on the caller. Bind a fresh local
   first.
2. **No closure may capture the model `m`.** It heap-copies the whole `MethanationModel`
   (1008 B per call). Bind the needed fields to locals.
3. **Aliased slots need the factor of 2** (§6), and testing against repo defaults will not
   catch its absence.
4. **Block iteration order must match** between `hess_structure` and `hess_values!` (§7).
5. **Tuple lengths must come from type parameters,** not runtime `nsp`/`nf`, or the tuples
   spill to memory: `nspecies_val`, `nfields_val`, and the `NTuple{N,Any}` signatures of
   `mix_cp` / `mix_conductivity` / `mixture_D` exist for this.
6. **`f_in` is state-dependent on $T$ inlet dofs** (§3.1) and contributes curvature.
