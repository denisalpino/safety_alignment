from typing import Dict, Any, List, Union

import numpy as np
import torch


# TODO: Typing for model param conflicts with weights extraction
def compute_feature_statistics(
    model: Any,
    activations_data: Union[np.ndarray, torch.Tensor, List],
    sample_size: int = 10000
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive statistics for Sparse Autoencoder (SAE) features.

    Analyzes activation patterns and encoder-decoder relationships to compute
    various feature metrics including density, bias, and reconstruction contributions.

    Parameters
    ----------
    model : torch.nn.Module
        SAE model with W_enc, W_dec, and optionally b_enc attributes
    activations_data : Union[np.ndarray, torch.Tensor, List]
        Activation data of shape (N, p) where N is number of samples, p is features
    sample_size : int, optional
        Maximum number of samples to use for computation, by default 10000

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing feature statistics:
        - density: Fraction of non-zero activations per feature
        - bias: Encoder bias values for each feature
        - de_dot: Dot product between encoder and decoder weights
        - de_cosine: Cosine similarity between encoder and decoder weights
        - de_frac_energy: Fraction of energy preserved in encoder-decoder loop
        - de_proj_coeff: Projection coefficient between encoder and decoder
        - recon_contrib: Relative contribution to reconstruction variance

    Examples
    --------
    >>> model = SparseAutoencoder()
    >>> activations = np.random.randn(5000, 256)
    >>> stats = compute_feature_statistics(model, activations)
    >>> print(stats['density'].shape)
    (256,)
    """
    model.eval()

    # Subsample data if larger than sample_size
    N = len(activations_data)
    if N > sample_size:
        idx = np.random.choice(N, size=sample_size, replace=False)
        acts = activations_data[idx]
    else:
        acts = activations_data

    # Convert to torch tensor if needed
    if isinstance(acts, (np.ndarray, list)):
        acts_t = torch.tensor(acts, dtype=torch.float32)
    else:
        acts_t = acts.float()

    with torch.no_grad():
        # Compute activation density (fraction of non-zero activations)
        density = (acts_t > 0).float().mean(dim=0).cpu().numpy()

        # Compute variance of latent activations
        var_z = acts_t.var(dim=0, unbiased=False).cpu().numpy()

        # Extract encoder and decoder weights
        encoder_eff = model.W_enc.data.cpu().numpy().copy()  # shape (p, n)
        decoder_eff = model.W_dec.data.cpu().numpy().copy()  # shape (n, p)

        p = encoder_eff.shape[0]
        de_dot = np.zeros(p, dtype=np.float32)
        de_cos = np.zeros(p, dtype=np.float32)
        de_frac_energy = np.zeros(p, dtype=np.float32)
        de_proj_coeff = np.zeros(p, dtype=np.float32)
        dec_norms = np.zeros(p, dtype=np.float32)

        # Compute encoder-decoder relationships for each feature
        for i in range(p):
            e = encoder_eff[i, :]  # Encoder weights for feature i
            d = decoder_eff[:, i]  # Decoder weights for feature i

            dot = float(np.dot(d, e))
            ne = np.linalg.norm(e)
            nd = np.linalg.norm(d)
            cos = dot / ((ne * nd) + 1e-12)
            frac_energy = cos * cos
            proj_coeff = dot / (ne * ne + 1e-12)

            de_dot[i] = dot
            de_cos[i] = cos
            de_frac_energy[i] = frac_energy
            de_proj_coeff[i] = proj_coeff
            dec_norms[i] = nd

        # Compute reconstruction contribution proportional to variance explained
        recon_contrib_raw = var_z * (dec_norms ** 2)
        recon_contrib = recon_contrib_raw / (recon_contrib_raw.sum() + 1e-12)

        # Extract encoder bias if available
        if model.b_enc is not None:
            bias = model.b_enc.data.cpu().numpy().astype(np.float32)
        else:
            bias = np.zeros(p, dtype=np.float32)

    return {
        'density': density,
        'bias': bias,
        'de_dot': de_dot,
        'de_cosine': de_cos,
        'de_frac_energy': de_frac_energy,
        'de_proj_coeff': de_proj_coeff,
        'recon_contrib': recon_contrib
    }
