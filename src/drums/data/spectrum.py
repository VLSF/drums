import numpy as np
import triangle as tr
import jax, jax.numpy as jnp
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
jax.config.update("jax_enable_x64", True) 

def local_matrices(xy):              # xy: (3, 2)
    x, y = xy[:, 0], xy[:, 1]
    b = jnp.roll(y, -1) - jnp.roll(y, -2)     # y_{i+1} - y_{i+2}
    c = jnp.roll(x, -2) - jnp.roll(x, -1)     # x_{i+2} - x_{i+1}
    A = 0.5 * ((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    Ke = (jnp.outer(b, b) + jnp.outer(c, c)) / (4*A)
    Me = A/12 * (jnp.ones((3, 3)) + jnp.eye(3))
    return Ke, Me

def get_drum_spectrum(drum_perimeter, N_eig=30, a=0.001):
    l = drum_perimeter.shape[0]
    S = np.array([(i, (i+1) % l) for i in range(l)])
    domain = dict(vertices=drum_perimeter, segments=S)

    mesh = tr.triangulate(domain, f'pq30a{a}')
    X = mesh['vertices']                 # (N, 2)
    T = mesh['triangles']                # (Ne, 3), CCW
    N = len(X)
    
    E = np.sort(np.vstack([T[:, [0,1]], T[:, [1,2]], T[:, [2,0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = np.unique(uniq[cnt == 1])
    free = np.setdiff1d(np.arange(N), bnd)

    Ke, Me = jax.jit(jax.vmap(local_matrices))(jnp.asarray(X)[T])   # (Ne,3,3) each
    Ke, Me = np.asarray(Ke), np.asarray(Me)
    
    rows = np.broadcast_to(T[:, :, None], Ke.shape).ravel()
    cols = np.broadcast_to(T[:, None, :], Ke.shape).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    M = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    
    Kf = K[free][:, free]
    Mf = M[free][:, free]
    
    lam1, U = eigsh(Kf, k=N_eig, M=Mf, sigma=0, which='LM')
    
    phi = np.zeros((N, N_eig))
    phi[free] = U
    return lam1, phi, X, T

if __name__ == "__main__":
    # Rectangle [0, p] x [0, q]: lambda_{mn} = pi^2 (m^2/p^2 + n^2/q^2)
    n_eig = 10
    ok = True
    for name, p, q in [("unit square", 1.0, 1.0), ("2x1 rectangle", 2.0, 1.0)]:
        perimeter = np.array([[0, 0], [p, 0], [p, q], [0, q]], dtype=float)
        lam, _, _, _ = get_drum_spectrum(perimeter, N_eig=n_eig, a=1e-4)
        lam = np.sort(lam)
        exact = np.sort([np.pi**2 * (m**2/p**2 + n**2/q**2)
                         for m in range(1, 20) for n in range(1, 20)])[:n_eig]
        close = np.allclose(lam, exact, rtol=1e-2)
        ok &= close
        print(f"{name}\n  computed: {lam}\n  expected: {exact}\n  allclose (rtol=1e-2): {close}")
    assert ok, "spectrum test failed"
