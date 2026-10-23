# RingTransformer: an invariant and equivariant transformer for closed polygons

This note describes the architecture implemented in `ring_transformer.py` (class `RingTransformer`). Indices in the formulas start from $1$, as in the mathematical description. The code uses indices starting from $0$.

## 1. Problem and symmetries

A sample is a closed polygon, given as a tuple of vertices $X = (x\_1, \dots, x\_n)$ with $x\_i \in \mathbb{R}^2$ and the same $n$ for all samples. The shape is formed by the edges $(x\_1, x\_2), (x\_2, x\_3), \dots, (x\_n, x\_1)$. Every sample also has a context vector $c \in \mathbb{R}^{m}$ of the same length $m$ for all samples.

The shape does not change under the following transformations of the tuple:

1. a rigid motion $x\_i \to O x\_i + b$ with $O \in O(2)$ (rotations and reflections) and a translation vector $b$;
2. a cyclic shift, for example $(x\_2, x\_3, \dots, x\_n, x\_1)$;
3. a reversion of the order, $(x\_n, x\_{n-1}, \dots, x\_1)$.

The architecture takes $(c, X)$ and returns two outputs:

1. an **invariant** vector $\widetilde{c} \in \mathbb{R}^{N\_{\text{inv}}}$, which does not change under any of the transformations above;
2. $n$ vectors $\Delta = (\delta\_1, \dots, \delta\_n)$, $\delta\_i \in \mathbb{R}^2$, which are **equivariant**: they are rotated and reflected like the vertices (translations do not change them, because they are built from the centred vertices), and they are permuted like the vertices under cyclic shifts and the reversion. They can be used as corrections to the vertices.

The construction first handles the rigid motions and the cyclic shifts with a transformer. The reversion is added at the end with two forward passes, see Section 7.

## 2. Removing shifts and rotations

The mean vector is subtracted,

$$
\widetilde{x}\_i = x\_i - \frac{1}{n} \sum\_{j=1}^{n} x\_j ,
$$

and the Gram matrix of the centred vertices is formed,

$$
G\_{ij} = \widetilde{x}\_i^{\top} \widetilde{x}\_j .
$$

A translation does not change $\widetilde{x}\_i$. A rotation or reflection $O$ changes it to $O \widetilde{x}\_i$ and does not change $G\_{ij}$, because $O^{\top} O = I$. So $G$ is invariant under all rigid motions. The vertices $\widetilde{x}\_i$ can be recovered from $G$ up to a common transformation from $O(2)$, so no information about the shape is lost. Note that $G$ has rank at most $2$.

## 3. Ring tokens and the context token

From the Gram matrix we form $n$ **ring tokens** of size $n$. Token $i$ is the row $i$ of $G$, cyclically shifted so that the diagonal element comes first,

$$
r\_i = \left( G\_{i,i},~ G\_{i,i+1},~ \dots,~ G\_{i,i+n-1} \right)^{\top} ,
$$

where the second index is taken modulo $n$. For $n = 3$ the tokens are

$$
r\_1 = \begin{pmatrix} G\_{11} \\ G\_{12} \\ G\_{13} \end{pmatrix}, \qquad
r\_2 = \begin{pmatrix} G\_{22} \\ G\_{23} \\ G\_{21} \end{pmatrix}, \qquad
r\_3 = \begin{pmatrix} G\_{33} \\ G\_{31} \\ G\_{32} \end{pmatrix} .
$$

If the vertices are shifted cyclically by $s$, so that $x'\_i = x\_{i+s}$, then $G'\_{ij} = G\_{i+s,j+s}$ and therefore $r'\_i = r\_{i+s}$. The tokens are permuted cyclically and their content does not change.

A shared linear layer maps every ring token from size $n$ to size $N\_f$ (the number of features, a parameter of the architecture). A separate linear layer maps the context $c$ from size $m$ to size $N\_f$. The result is the sequence of $n + 1$ tokens

$$
\left( t\_c, t\_{r\_1}, \dots, t\_{r\_n} \right) ,
$$

which is processed by $L$ transformer layers.

## 4. Attention with a circulant bias

The layers are standard pre-norm transformer layers, except for the attention. For every head $h$ the attention logits are

$$
A^{(h)}\_{ab} = \frac{q\_a^{\top} k\_b}{\sqrt{d}} + B^{(h)}\_{ab} , \qquad a, b = 0, 1, \dots, n ,
$$

where $d$ is the size of a head, the index $0$ is the context token, and the indices $1, \dots, n$ are the ring tokens. The attention weights are the softmax of $A^{(h)}$ over $b$. There is no mask: all tokens attend to all tokens. The learnable bias has the following structure (for $n = 3$)

$$
B^{(h)} = \begin{pmatrix}
\gamma & \alpha & \alpha & \alpha \\
\beta & b\_0 & b\_1 & b\_2 \\
\beta & b\_2 & b\_0 & b\_1 \\
\beta & b\_1 & b\_2 & b\_0
\end{pmatrix} .
$$

For general $n$ the ring block is the circulant matrix $B\_{ij} = b\_{(j-i) \bmod n}$ for the ring tokens $i, j = 1, \dots, n$. The first row (context query) is $\gamma$ for the context key and $\alpha$ for all ring keys, and the first column (context key) is $\beta$ for all ring queries.

So every head of every layer has $n + 3$ learnable scalars: $\gamma$, $\alpha$, $\beta$ and $b\_0, \dots, b\_{n-1}$.

**Why this bias.** The standard attention without a bias does not depend on the order of the tokens, so it commutes with all permutations of the ring tokens. Then the attention could not use the position of a token in the ring, for example which token is the neighbour of which. The circulant bias breaks the symmetry from all permutations down to the cyclic shifts: shifting the ring tokens cyclically by $s$ leaves the circulant matrix unchanged, because $b\_{(j+s)-(i+s)} = b\_{j-i}$, and the other entries of $B$ are constant. As a consequence the attention commutes with cyclic shifts, and does not commute with an arbitrary permutation.

All the other operations of a layer (layer normalisation, the MLP, the residual connections) act on every token separately and in the same way, so they also commute with the permutation of tokens. For a layer

$$
\begin{split}
x &\leftarrow x + \text{Attention}\left(\text{LayerNorm}(x)\right) , \\
x &\leftarrow x + \text{MLP}\left(\text{LayerNorm}(x)\right) ,
\end{split}
$$

with an MLP that has one hidden layer of size $4 N\_f$ and the GELU nonlinearity. After the last layer a final layer normalisation is applied to all tokens.

## 5. Outputs

**Invariant output.** A linear layer maps the final context token to the size $N\_{\text{inv}}$,

$$
\widetilde{c} = W\_{\text{inv}}~ t\_c + b\_{\text{inv}} .
$$

The context token is invariant to the rigid motions, because the input is built from $G$ and $c$. It is also invariant to the cyclic shifts: every layer commutes with a cyclic shift of the ring tokens (Section 4), and the context token itself is not moved by the shift. Hence $\widetilde{c}$ is invariant.

**Equivariant output.** A common linear layer maps each final ring token $t\_{r\_i}$ from size $N\_f$ to size $n$. We denote the result by $f^{(i)} = \left( f^{(i)}\_1, \dots, f^{(i)}\_n \right)$. The vector $\delta\_i$ is a linear combination of the centred vertices, which are cyclically shifted so that the vertex $i$ comes first,

$$
\delta\_i = \sum\_{k=1}^{n} f^{(i)}\_k~ \widetilde{x}\_{i+k-1} ,
$$

with the index taken modulo $n$. For $n = 3$ this is

$$
\begin{split}
\delta\_1 &= f^{(1)}\_1 \widetilde{x}\_1 + f^{(1)}\_2 \widetilde{x}\_2 + f^{(1)}\_3 \widetilde{x}\_3 , \\
\delta\_2 &= f^{(2)}\_1 \widetilde{x}\_2 + f^{(2)}\_2 \widetilde{x}\_3 + f^{(2)}\_3 \widetilde{x}\_1 , \\
\delta\_3 &= f^{(3)}\_1 \widetilde{x}\_3 + f^{(3)}\_2 \widetilde{x}\_1 + f^{(3)}\_3 \widetilde{x}\_2 .
\end{split}
$$

The coefficients $f^{(i)}\_k$ are invariant to the rigid motions, because they come from invariant tokens. Hence under $\widetilde{x}\_i \to O \widetilde{x}\_i$ the vector $\delta\_i$ becomes $O \delta\_i$, which is the equivariance to rotations and reflections. Under a cyclic shift by $s$ the token $r\_i$ becomes $r\_{i+s}$, so $f'^{(i)} = f^{(i+s)}$, and the shifted vertices give $\delta'\_i = \delta\_{i+s}$, which is the equivariance to cyclic shifts. The linear layer may have a bias: it is the same for all tokens, so it keeps the equivariance.

The mean vector can be added to $\delta\_i$ outside of the architecture if needed.

## 6. The forward pass

The forward pass $f(c, X)$ returns the pair $\left( \widetilde{c}, \Delta \right)$ with $\widetilde{c} \in \mathbb{R}^{N\_{\text{inv}}}$ and $\Delta \in \mathbb{R}^{n \times 2}$. By construction it is invariant and equivariant with respect to rigid motions and cyclic shifts, but **not** with respect to the reversion of the order. Under the reversion, ring token $i$ becomes $\left( G\_{i,i}, G\_{i,i-1}, \dots \right)$, the row read in the opposite direction, and this is none of the original tokens.

## 7. Taking the reversion into account

The reversion is added with two forward passes, whose results are combined,

$$
\begin{split}
\widetilde{c}\_1, \Delta\_1 &\leftarrow f(c, X) , \\
\widetilde{c}\_2, \Delta\_2 &\leftarrow f(c, X^{\text{rev}}) , \\
\widetilde{c}\_{\text{flip}} &= \frac{1}{2} \left( \widetilde{c}\_1 + \widetilde{c}\_2 \right) , \\
\Delta\_{\text{flip}} &= \frac{1}{2} \left( \Delta\_1 + \Delta\_2^{\text{rev}} \right) ,
\end{split}
$$

where $X^{\text{rev}} = (x\_n, \dots, x\_1)$ and $\Delta^{\text{rev}}$ denote the sequences with the reversed order of the indices (`X[::-1]` in the code). The reversal of $\Delta\_2$ aligns it with the original order of the vertices.

If the input is reversed, $X' = X^{\text{rev}}$, the roles of the two passes are exchanged: the first pass gives $\widetilde{c}\_2, \Delta\_2$ and the second pass gives $\widetilde{c}\_1, \Delta\_1$. So $\widetilde{c}\_{\text{flip}}$ is unchanged, and $\Delta\_{\text{flip}}$ becomes $\frac{1}{2} \left( \Delta\_2 + \Delta\_1^{\text{rev}} \right) = \Delta\_{\text{flip}}^{\text{rev}}$. The pair is invariant and equivariant with respect to all the transformations of Section 1. The combination of the two passes keeps the invariance to the rigid motions and the cyclic shifts, since each pass has it.

## 8. Implementation

The implementation is in `ring_transformer.py`:

- `RingAttention` is a multi-head attention with the learnable bias of Section 4. It is written from scratch, because `equinox.nn.MultiheadAttention` can only mask the logits and cannot add a learnable bias.
- `Block` is a pre-norm transformer block with `RingAttention`.
- `RingTransformer(n, m, N_f, N_inv, key, L, n_heads=4, mlp_ratio=4)` is the full model. Calling `model(c, X)` gives the forward pass $f(c, X)$ of Section 6, and `model.forward_flip(c, X)` gives the pass of Section 7.

The model processes one sample ($c$ of shape `(m,)` and $X$ of shape `(n, 2)`). Use `jax.vmap` for batches.

Choices which are not fixed by the description above:

- The bias is added to the logits after the division by $\sqrt{d}$, separately for every layer and every head.
- The bias parameters $\gamma, \alpha, \beta, b\_j$ are initialised with zeros.
- The model uses pre-norm layers with a final layer normalisation, as in `DrumTransformer`.
- The input is not normalised inside the model. The Gram matrix entries grow like the square of the size of the polygon, so a normalisation (outside of the model) may be needed.
- $\Delta$ is invariant to translations of $X$. The mean vector is not added to it.

The number of parameters is

$$
n N\_f + N\_f + m N\_f + N\_f + L \cdot P\_{\text{block}} + 2 N\_f + N\_f N\_{\text{inv}} + N\_{\text{inv}} + N\_f n + n ,
$$

with $P\_{\text{block}} = 4 N\_f + 4 \left( N\_f^2 + N\_f \right) + H (n + 3) + 8 N\_f^2 + 5 N\_f$ for $H$ heads and the MLP ratio $4$.

## 9. Tests

`python src/drums/models/ring_transformer.py` runs the tests in double precision (in single precision a batched and a single evaluation can differ by $10^{-3}$ on a GPU). The learnable bias is set to random values for the tests, because with zeros it would not be tested. The tests check that:

- for the forward pass $f$, $\widetilde{c}$ is invariant and $\Delta$ is equivariant under a rotation, a reflection, a translation, a cyclic shift and their combination;
- for the pass with the flip, the same holds also for the reversion and the reversion combined with all the other transformations;
- the plain forward pass is not symmetric under the reversion and under a general permutation, and a bias that is not circulant breaks the cyclic symmetry (so the tests can fail);
- the batched evaluation with `jax.vmap` agrees with the single evaluation, the gradients are finite and reach the bias, and the number of parameters agrees with the formula above.

The symmetries hold to about $10^{-15}$ in double precision.
