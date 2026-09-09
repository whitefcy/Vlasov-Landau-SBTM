import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx
import orbax.checkpoint as ocp
import os
import jax.lax as lax

class MLPScoreModel(nnx.Module):
    """MLP-based implementation of a score model."""
    
    def save(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        _, state = nnx.split(self)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path + '/state', state, **kwargs)
        return None

    def load(self, path):
        path += '/state'
        graphdef, abstract_state = nnx.split(self)
        checkpointer = ocp.StandardCheckpointer()
        state_restored = checkpointer.restore(path, abstract_state)

        # Get a new model instance with the restored weights
        restored = nnx.merge(graphdef, state_restored)

        # In-place update of known fields
        self.layers = restored.layers
        self.final_layer = restored.final_layer
        return None

    
    def __init__(self, dx, dv, hidden_dims=[128, 128], activation=nnx.soft_sign, seed=0, dtype=jnp.float32):
        """Initialize MLP score model.
        
        Args:
            dx: Dimension of position (x) input
            dv: Dimension of velocity (v) input
            hidden_dims: Sequence of hidden dimensions
            activation: Activation function
            seed: Random seed for initialization
            dtype: Data type for all layers (e.g., jnp.float32 or jnp.float64)
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dtype = dtype
        rngs = nnx.Rngs(seed)
        
        # Initialize layers immediately
        self.layers = []
        input_dim = dx + dv  # Concatenated input dimensions
        
        for hidden_dim in self.hidden_dims:
            self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs, dtype=self.dtype))
            input_dim = hidden_dim
        
        self.final_layer = nnx.Linear(input_dim, dv, rngs=rngs, dtype=self.dtype)
        # Flax >=0.12 requires containers of layers to be marked as data.
        if hasattr(nnx, 'data'):
            self.layers = nnx.data(self.layers)
    
    def __call__(self, x, v):
        """Compute the score using an MLP network.
        
        Args:
            x: Position array of shape (batch_size, dx) or (..., dx)
            v: Velocity array of shape (batch_size, dv) or (..., dv)
            
        Returns:
            Score vector of shape (batch_size, dv) or (..., dv),
            matching the batch dimensions of the inputs
        """
        # Concatenate x and v along the last dimension
        if x.ndim < v.ndim:
            x = x[:, None]
        inputs = jnp.concatenate([x, v], axis=-1)
        
        # Pass through MLP layers
        h = inputs
        for layer in self.layers:
            h = self.activation(layer(h))
        
        # Final layer to produce the score
        outputs = self.final_layer(h)
        
        return outputs

class ResNetScoreModel(nnx.Module):
    """ResNet-based implementation of a score model."""
    
    def __init__(self, mlp, seed=0, dtype=jnp.float32):
        """Initialize ResNet score model using an MLP with residual connections.
        
        Args:
            mlp: An MLPScoreModel instance to use as the base network
            seed: Random seed for initialization
            dtype: Data type for all layers (e.g., jnp.float32 or jnp.float64)
        """
        self.mlp = mlp
        self.activation = mlp.activation
        self.rngs = nnx.Rngs(seed)
        self.dtype = dtype
        
        # Create projections for residual connections
        self.projections = []
        
        # Get the dimensions from the first layer in the MLP
        if len(self.mlp.layers) > 0:
            input_dim = self.mlp.layers[0].kernel.shape[0]  # Input dimension of first layer
            
            for layer in self.mlp.layers:
                out_dim = layer.kernel.shape[1]  # Output dimension of the layer
                
                # If dimensions don't match, create a projection
                if input_dim != out_dim:
                    projection = nnx.Linear(input_dim, out_dim, rngs=self.rngs, dtype=self.dtype)
                    self.projections.append(projection)
                else:
                    self.projections.append(None)
                
                input_dim = out_dim
    
    def __call__(self, x, v):
        """Compute the score using a ResNet architecture.
        
        Args:
            x: Position array of shape (batch_size, dx) or (..., dx)
            v: Velocity array of shape (batch_size, dv) or (..., dv)
            
        Returns:
            Score vector of shape (batch_size, dv) or (..., dv),
            matching the batch dimensions of the inputs
        """
        # Concatenate x and v along the last dimension
        if x.ndim < v.ndim:
            x = x[:, None]
        inputs = jnp.concatenate([x, v], axis=-1)
        
        # Apply ResNet layers with skip connections
        h = inputs
        
        # Apply each layer with its corresponding projection/skip connection
        for i, (layer, projection) in enumerate(zip(self.mlp.layers, self.projections)):
            layer_output = self.activation(layer(h))
            
            # Apply projection if dimensions don't match, otherwise add directly
            if projection is not None:
                h = layer_output + projection(h)
            else:
                h = layer_output + h
        
        # Final layer to produce the score
        outputs = self.mlp.final_layer(h)
        
        return outputs

"KDE score model for 1D space"
def _silverman_bandwidth(v, eps=1e-12):
        n, dv = v.shape
        sigma = jnp.std(v, axis=0, ddof=1) + eps
        return sigma * n ** (-1.0 / (dv + 4.0))  # (dv,)

@partial(jax.jit, static_argnames=['max_ppc'])
def _score_blob_local_impl(x, v, cells, eta, eps=1e-12, hv=None, max_ppc=4096):
    if x.ndim == 2:
        x = x[:, 0]
    if hv is None:
        hv = _silverman_bandwidth(v, eps)

    n, dv = v.shape
    M = cells.size
    L = eta * M
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv ** 2

    idx = jnp.floor(x / eta).astype(jnp.int32) % M
    order = jnp.argsort(idx)
    x_s = x[order]
    v_s = v[order]
    idx_s = idx[order]

    counts = jnp.bincount(idx_s, length=M)
    cell_ofs = jnp.cumsum(
        jnp.concatenate([jnp.array([0], dtype=jnp.int32), counts[:-1]])
    )

    Xc = jnp.zeros((M, max_ppc), x.dtype)
    Vc = jnp.zeros((M, max_ppc, dv), v.dtype)
    maskc = jnp.zeros((M, max_ppc), x.dtype)
    idx_map = -jnp.ones((M, max_ppc), jnp.int32)
    ar_ppc = jnp.arange(max_ppc, dtype=jnp.int32)

    def fill_cell(c, carry):
        Xc, Vc, maskc, idx_map = carry
        cnt = counts[c]
        base = cell_ofs[c]
        valid = ar_ppc < cnt
        gidx = base + ar_ppc
        gidx = jnp.where(valid, gidx, 0)
        Xc = Xc.at[c].set(jnp.where(valid, x_s[gidx], 0.0))
        Vc = Vc.at[c].set(jnp.where(valid[:, None], v_s[gidx], 0.0))
        maskc = maskc.at[c].set(valid.astype(x.dtype))
        idx_map = idx_map.at[c].set(jnp.where(valid, gidx, -1))
        return Xc, Vc, maskc, idx_map

    Xc, Vc, maskc, idx_map = lax.fori_loop(
        0, M, fill_cell, (Xc, Vc, maskc, idx_map)
    )

    Uc = Vc * inv_hv
    U2c = jnp.sum(Uc * Uc, axis=-1, keepdims=True)

    Zc = jnp.zeros((M, max_ppc, 1), v.dtype)
    Mc = jnp.zeros((M, max_ppc, dv), v.dtype)

    def body_cell(c, carry):
        Zc, Mc = carry
        Xi = Xc[c]                    # (max_ppc,)
        Vi = Vc[c]                    # (max_ppc,dv)
        Ui = Uc[c]
        Ui2 = U2c[c]                  # (max_ppc,1)
        mask_i = maskc[c][:, None]    # (max_ppc,1)

        c0 = (c - 1) % M
        c1 = c
        c2 = (c + 1) % M
        Xj = jnp.concatenate([Xc[c0], Xc[c1], Xc[c2]], axis=0)        # (3*max_ppc,)
        Vj = jnp.concatenate([Vc[c0], Vc[c1], Vc[c2]], axis=0)        # (3*max_ppc,dv)
        Uj = jnp.concatenate([Uc[c0], Uc[c1], Uc[c2]], axis=0)
        Uj2 = jnp.concatenate([U2c[c0], U2c[c1], U2c[c2]], axis=0)    # (3*max_ppc,1)
        mask_j = jnp.concatenate(
            [maskc[c0], maskc[c1], maskc[c2]], axis=0
        )[:, None]                                                     # (3*max_ppc,1)

        dx = Xi[:, None] - Xj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.maximum(0.0, 1.0 - jnp.abs(dx) / eta)               # hat in x

        G = Ui @ Uj.T
        K = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2.T)

        mask = mask_i * mask_j.T
        w = (psi * K + eps) * mask

        Z_local = jnp.sum(w, axis=1, keepdims=True) * mask_i
        M_local = (w @ Vj) * mask_i

        Zc = Zc.at[c].set(Z_local)
        Mc = Mc.at[c].set(M_local)
        return Zc, Mc

    Zc, Mc = lax.fori_loop(0, M, body_cell, (Zc, Mc))

    idx_flat = idx_map.reshape(-1)
    Z_flat = Zc.reshape(-1, 1)
    M_flat = Mc.reshape(-1, dv)
    valid = idx_flat >= 0
    idx_valid = jnp.where(valid, idx_flat, 0)
    Z_contrib = Z_flat * valid[:, None]
    M_contrib = M_flat * valid[:, None]

    Zs = jnp.zeros((n, 1), v.dtype)
    Ms = jnp.zeros((n, dv), v.dtype)
    Zs = Zs.at[idx_valid].add(Z_contrib)
    Ms = Ms.at[idx_valid].add(M_contrib)

    inv_order = jnp.empty_like(order)
    inv_order = inv_order.at[order].set(jnp.arange(n))

    Z = Zs[inv_order]
    M = Ms[inv_order]
    
    Z_safe = jnp.where(Z > 0, Z, eps)
    mu = M / Z_safe
    jax.debug.print("max_ppc={max_ppc}, max_count={mc}", max_ppc=max_ppc, mc=jnp.max(counts))
    return (mu - v) * inv_hv2

# this is ~11 times faster than score_blob_blocked with n=1e5 and M=50
def score_blob(x, v, cells, eta, eps=1e-12, hv=None):
    if hv is None:
        hv = _silverman_bandwidth(v, eps)

    if x.ndim == 2:
        x1d = x[:, 0]
    else:
        x1d = x
    M = cells.size
    idx = jnp.floor(x1d / eta).astype(jnp.int32) % M
    counts = jnp.bincount(idx, length=M)
    max_count = int(jax.device_get(jnp.max(counts)))
    m = max(1, max_count)
    max_ppc = ((m + 99) // 100) * 100  # next multiple of 100 >= m

    return _score_blob_local_impl(x, v, cells, eta, eps, hv, max_ppc)
