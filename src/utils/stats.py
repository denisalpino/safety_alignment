import math
from collections import Counter
from typing import Dict, Any, List, Tuple, Union

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


def get_top_tokens(
    prompts: List[List[str]],
    activations: Union[List[float], np.ndarray],
    top_k: int = 10,
    method: str = 'tfidf',
    norm: bool = True
) -> List[Tuple[str, float]]:
    """
    Find the most relevant tokens for a neuron based on activation patterns.

    This function analyzes token activations across multiple text prompts to identify
    which tokens are most relevant to a specific neuron using various scoring methods.

    Parameters
    ----------
    prompts : List[List[str]]
        List of tokenized text prompts where each prompt is a list of tokens
    activations : Union[List[float], np.ndarray]
        Array of activation values corresponding to each token across all prompts
    top_k : int, optional
        Number of top tokens to return, by default 10
    method : str, optional
        Scoring method: 'tfidf', 'mean_activation', or 'bm25', by default 'tfidf'
    norm : bool, optional
        Whether to normalize relevance scores to [0, 1] range, by default True

    Returns
    -------
    List[Tuple[str, float]]
        List of (token, score) tuples for top-K tokens, sorted by descending relevance

    Raises
    ------
    ValueError
        If method is not one of supported scoring methods
        If prompts and activations length mismatch occurs

    Examples
    --------
    >>> prompts = [["hello", "world"], ["test", "token"]]
    >>> activations = [0.1, 0.8, 0.3, 0.9]
    >>> top_tokens = get_top_tokens(prompts, activations, top_k=2)
    >>> print(top_tokens)
    [("token", 1.0), ("world", 0.888)]
    """
    activations = np.array(activations)

    # Validate scoring method
    valid_methods = {'tfidf', 'mean_activation', 'bm25'}
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")

    # Process only prompts that have corresponding activations
    total_tokens = 0
    valid_prompts = []
    for prompt in prompts:
        if total_tokens + len(prompt) <= len(activations):
            valid_prompts.append(prompt)
            total_tokens += len(prompt)
        else:
            break

    if not valid_prompts:
        return []

    # Build token activation mappings and document frequencies
    token_activations = {}
    token_document_freq = Counter()
    current_index = 0

    for prompt in valid_prompts:
        prompt_tokens = set()  # Track unique tokens per document for DF calculation

        for token in prompt:
            if token not in token_activations:
                token_activations[token] = []
            token_activations[token].append(activations[current_index])
            prompt_tokens.add(token)
            current_index += 1

        for token in prompt_tokens:
            token_document_freq[token] += 1

    total_documents = len(valid_prompts)

    # Calculate relevance scores using specified method
    token_scores = {}
    for token, acts in token_activations.items():
        acts_array = np.array(acts)
        avg_activation = np.mean(acts_array)

        if method == 'mean_activation':
            score = avg_activation

        elif method == 'tfidf':
            tf = len(acts)  # Term frequency - total occurrences
            df = token_document_freq[token]
            idf = math.log((total_documents + 1) / (df + 1)) + 1  # Smoothing
            score = avg_activation * tf * idf

        elif method == 'bm25':
            k1, b = 1.2, 0.75
            avg_doc_length = total_tokens / total_documents
            tf = len(acts)
            df = token_document_freq[token]

            idf = math.log((total_documents - df + 0.5) / (df + 0.5) + 1)
            denominator = tf + k1 * (1 - b + b * len(valid_prompts[0]) / avg_doc_length)
            score = avg_activation * idf * (tf * (k1 + 1)) / denominator

        token_scores[token] = score

    # Sort and select top-K tokens
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:top_k]

    # Normalize scores if requested
    if norm and top_tokens:
        max_score = top_tokens[0][1]
        if max_score > 0:
            return [(token, score / max_score) for token, score in top_tokens]

    return top_tokens


def pretty_scale(
    activations: Union[List, np.ndarray, torch.Tensor],
    neuron_idx: int,
    min_val: float = 0.1,
    max_val: float = 0.9
):
    """
    Normalize non-zero activations of a specific neuron to a specified range while preserving zeros.

    This function processes activations for a given neuron, keeping zero values as zero
    and normalizing non-zero values to the range [min_val, max_val]. All non-zero values
    are assumed to be positive; negative values will raise an exception.

    Parameters
    ----------
    activations : Union[List, numpy.ndarray, torch.Tensor]
        2D array of activations with shape (n_samples, n_neurons)
    neuron_idx : int
        Index of the neuron to normalize
    min_val : float, optional
        Minimum value of the normalized range (default: 0.1)
    max_val : float, optional
        Maximum value of the normalized range (default: 0.9)

    Returns
    -------
    numpy.ndarray
        Normalized activations for the specified neuron, where:
        - Zero values remain zero
        - Non-zero values are scaled to [min_val, max_val]

    Raises
    ------
    ValueError
        If activations contain negative values
        If min_val >= max_val
        If neuron_idx is out of bounds

    Examples
    --------
    >>> activations = np.array([[0.0, 1.0, 2.0], [0.5, 0.0, 3.0]])
    >>> pretty_scale(activations, neuron_idx=0)
    array([0.1, 0.5])  # 0.0→0.0, 0.5→0.5 (normalized between 0.1 and 0.9)
    """
    # Input validation
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")

    # Convert to nump array if needed
    if isinstance(activations, (torch.Tensor, list)):
        activations = np.array(activations)

    if neuron_idx < 0 or neuron_idx >= activations.shape[1]:
        raise ValueError(f"neuron_idx {neuron_idx} is out of bounds for activations shape {activations.shape}")

    # Extract specific neuron activations
    neuron = activations[:, neuron_idx]

    # Check for negative values
    if np.any(neuron < 0):
        raise ValueError("Activations contain negative values - only non-negative values are supported")

    # Create mask for non-zero values
    non_zero_mask = neuron > 0

    # If no non-zero values, return zeros
    if not np.any(non_zero_mask):
        return neuron

    # Extract non-zero values
    non_zero_vals = neuron[non_zero_mask]

    # Handle case where all non-zero values are the same
    if non_zero_vals.min() == non_zero_vals.max():
        normalized_non_zero = np.full_like(non_zero_vals, (min_val + max_val) / 2)
    else:
        # Normalize non-zero values to [min_val, max_val]
        normalized_non_zero = (non_zero_vals - non_zero_vals.min()) / (non_zero_vals.max() - non_zero_vals.min())
        normalized_non_zero = normalized_non_zero * (max_val - min_val) + min_val

    # Create result array and assign normalized values
    result = np.zeros_like(neuron)
    result[non_zero_mask] = normalized_non_zero

    return result
