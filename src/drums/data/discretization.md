# Discretization of the drum spectrum

This note describes how `get_drum_spectrum` in `spectrum.py` turns the shape of a drum into a list of eigenvalues.

## 1. Problem

A drum is a polygon $\Omega \subset \mathbb{R}^2$ with a fixed (clamped) boundary. Its spectrum is the set of numbers $\lambda$ for which the Dirichlet eigenvalue problem has a non-trivial solution:

$$
-\Delta u = \lambda u \quad \text{in } \Omega, \qquad u = 0 \quad \text{on } \partial\Omega .
$$

Multiplying by a test function $v$ that vanishes on $\partial\Omega$ and integrating by parts gives the weak form: find $\lambda$ and $u \neq 0$ such that

$$
\int_\Omega \nabla u \cdot \nabla v ~ dx = \lambda \int_\Omega u v ~ dx
\quad \text{for all } v \in H^1\_0(\Omega).
$$

## 2. Mesh

The polygon is given by its vertices $p\_1, \dots, p\_n$, joined by the segments $(p\_i, p\_{i+1})$ with the last vertex joined to the first. This closed boundary is passed to the mesh generator Triangle with the switches `pq30a{a}`:

- `p` triangulates the polygon given as a set of vertices and segments,
- `q30` keeps all angles at least $30^\circ$, so that the elements are not degenerate,
- `a{a}` bounds the area of every triangle by `a`.

The result is a set of nodes $X \in \mathbb{R}^{N \times 2}$ and a list of triangles $T$, where each row holds three node indices in counter-clockwise order.

Boundary nodes are found from the connectivity: an edge that belongs to exactly one triangle lies on $\partial\Omega$. All nodes on such edges form the set $\mathcal{B}$, and the remaining nodes are the free (interior) nodes $\mathcal{F}$.

## 3. Linear (P1) elements

On each triangle the unknown $u$ is approximated by a linear function. Globally,

$$
u\_h(x, y) = \sum\_{j=1}^{N} u^j \varphi^j(x, y),
$$

where $\varphi^j$ is the hat function of node $j$: it is continuous, linear on every triangle, equal to $1$ at node $j$ and $0$ at all other nodes. The coefficient $u^j$ is the value of $u\_h$ at node $j$.

Dirichlet conditions are imposed by setting $u^j = 0$ for $j \in \mathcal{B}$, so only the free nodes carry unknowns. Taking $v = \varphi^i$ for $i \in \mathcal{F}$ in the weak form gives the generalized matrix eigenvalue problem

$$
K u = \lambda M u,
$$

with the stiffness and mass matrices

$$
K^{ij} = \int\_\Omega \nabla\varphi^i \cdot \nabla\varphi^j ~ dx,
\qquad
M^{ij} = \int\_\Omega \varphi^i \varphi^j ~ dx,
\qquad i, j \in \mathcal{F}.
$$

## 4. Local matrices on one triangle

Consider a triangle with vertices $(x^0, y^0)$, $(x^1, y^1)$, $(x^2, y^2)$, listed counter-clockwise. Its area is

$$
A = \tfrac12 \big[ (x^1 - x^0)(y^2 - y^0) - (x^2 - x^0)(y^1 - y^0) \big] > 0 .
$$

On the triangle only three hat functions are non-zero. Restricted to it they are the barycentric coordinates $N^0, N^1, N^2$, which are linear and satisfy $N^i = 1$ at vertex $i$ and $N^i = 0$ at the other two vertices. Their gradients are constant:

$$
\nabla N^i = \frac{1}{2A} \begin{pmatrix} b^i \\ c^i \end{pmatrix},
\qquad
b^i = y^{i+1} - y^{i+2},
\qquad
c^i = x^{i+2} - x^{i+1},
$$

with indices taken modulo $3$. (In the code, `jnp.roll` computes these shifted differences.) The gradient $\nabla N^i$ is perpendicular to the edge opposite to vertex $i$, and its length is $1$ divided by the height of the triangle over that edge.

**Local stiffness matrix.** Since the gradients are constant, the integral is just the gradient product times the area:

$$
K^e\_{ij} = \int\_e \nabla N^i \cdot \nabla N^j ~ dx
= A ~ \frac{b^i b^j + c^i c^j}{4A^2}
= \frac{b^i b^j + c^i c^j}{4A}.
$$

**Local mass matrix.** For barycentric coordinates there is the exact integration formula

$$
\int\_e (N^0)^{\alpha} (N^1)^{\beta} (N^2)^{\gamma} ~ dx
= 2A ~ \frac{\alpha! ~ \beta! ~ \gamma!}{(\alpha + \beta + \gamma + 2)!}.
$$

For the diagonal entries ($\alpha = 2$) this gives $2A \cdot 2/4! = A/6$, and for the off-diagonal entries ($\alpha = \beta = 1$) it gives $2A/4! = A/12$. In compact form

$$
M^e = \frac{A}{12}
\begin{pmatrix}
2 & 1 & 1 \\
1 & 2 & 1 \\
1 & 1 & 2
\end{pmatrix}
= \frac{A}{12} \big( \mathbf{1}\mathbf{1}^{\top} + I \big),
$$

which is what `local_matrices` returns. It is evaluated for all triangles at once with `jax.vmap` and `jax.jit`, in double precision.

## 5. Assembly

The global matrices are sums of the local contributions. If triangle $e$ has nodes $T\_{e0}, T\_{e1}, T\_{e2}$, then

$$
K = \sum\_e \sum\_{i, j = 0}^{2} K^e\_{ij} ~ E\_{T\_{ei}, T\_{ej}},
$$

where $E\_{rs}$ is the matrix with a single $1$ in position $(r, s)$. The same holds for $M$. In the code, the triples (row, column, value) of all triangles are collected in a COO matrix, and entries that share a position are summed when the matrix is converted to CSR. Both matrices are sparse and symmetric, and $M$ is positive definite.

## 6. Boundary conditions and the eigenvalue solver

Rows and columns of the boundary nodes are removed, which is the same as setting $u^j = 0$ there:

$$
K\_f = K[\mathcal{F}, \mathcal{F}], \qquad M\_f = M[\mathcal{F}, \mathcal{F}].
$$

The smallest `N_eig` eigenvalues of $K\_f u = \lambda M\_f u$ are computed with `scipy.sparse.linalg.eigsh` in shift-invert mode with shift $\sigma = 0$. The solver then works with $(K\_f - \sigma M\_f)^{-1} M\_f$, whose largest eigenvalues $1/(\lambda - \sigma)$ correspond to the smallest $\lambda$. The eigenvalues are not guaranteed to come back sorted, so sort them before comparing spectra.

The function returns

- `lam1`: the eigenvalues $\lambda\_k$,
- `phi`: an array of shape $(N, \texttt{N\_eig})$ with the eigenfunctions at all nodes, zero at the boundary nodes (the eigenvectors are normalized so that $u^{\top} M\_f u = 1$),
- `X`, `T`: the nodes and the triangles of the mesh.

## 7. Accuracy

With linear elements the computed eigenvalues approximate the exact ones from above, and the error decreases like $O(h^2)$, where $h$ is the mesh size (controlled by `a`). The error grows with the index of the eigenvalue because higher modes oscillate faster. For example, the eigenvalues of the rectangle $[0, p] \times [0, q]$ are known exactly,

$$
\lambda\_{mn} = \pi^2 \left( \frac{m^2}{p^2} + \frac{n^2}{q^2} \right), \qquad m, n = 1, 2, \dots,
$$

and the tests at the bottom of `spectrum.py` compare the computed spectrum with this formula.
