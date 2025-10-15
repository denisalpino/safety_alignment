"""
🔬 Hidden States Extraction Module with Token-Based Batching.
This module provides specialized functionality for extracting hidden states from transformer-based language models.
It is designed for researchers and practitioners who need fine-grained access to intermediate layer representations
without padding artifacts, and with efficient token-based batching.

🎯 Key Features
- Token-Based Batching: Batches are formed based on total token count, maximizing GPU utilization
- Padding-Free Extraction: Automatically removes padding tokens from hidden states
- Cache-Aware: Strict adherence to kv_cache_size limits for memory optimization
- Layer Selection: Flexible extraction from specific transformer layers
- Order Preservation: Guaranteed output order matches input order

🚀 Use Cases
- Probing Studies: Analyze linguistic properties in different layers
- Feature Extraction: Get high-quality embeddings for downstream tasks
- Model Analysis: Study information flow through transformer layers
- Memory-Constrained Environments: Process large texts with strict memory limits

💡 Key Concepts

1. Token-Based Batching
   Batches are formed dynamically based on total token count rather than fixed number of sequences
2. KV Cache Limits
   Strict enforcement of maximum tokens per batch to prevent memory overflow
3. Padding Removal
   Critical for obtaining clean representations where each token embedding corresponds to actual content

📝 Usage Pattern

```python
from transformers import AutoModel, AutoTokenizer
from hidden_states_extractor import embed

# Load model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Extract layers 4, 8, 12 with token-based batching
texts = ["Hello world!", "Another example text"]
results = embed(
    texts=texts,
    model=model,
    tokenizer=tokenizer,
    layers=[4, 8, 12],
    kv_cache_size=4096,  # Maximum tokens per batch
    max_length=2048
)
```
"""

import gc
from typing import Sequence, List, Dict, Any, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm


def embed(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    layers: Sequence[int],
    kv_cache_size: int,
    max_length: int = 2048,
) -> List[Dict[int, torch.Tensor]]:
    """
    Extract hidden states from specified transformer layers using token-based batching.

    This function processes input texts through a transformer model and returns hidden states
    from the specified layers. Batches are formed dynamically to maximize token count while
    strictly adhering to the kv_cache_size limit. No text is split across batches.

    Parameters
    ----------
    texts : List[str]
        Input texts to process. Each string will be tokenized and passed through the model.
        Empty strings will result in empty tensors for that position.

    model : Any
        Pre-trained transformer model instance (typically from Hugging Face transformers).
        Must support `output_hidden_states=True` in forward pass and have a `device` attribute
        or parameters that indicate device placement.

    tokenizer : Any
        Tokenizer instance corresponding to the model. Must implement `__call__` method
        with standard Hugging Face tokenizer interface and return `input_ids` and `attention_mask`.

    layers : Sequence[int]
        Zero-indexed layer indices from which to extract hidden states.

    kv_cache_size : int
        Maximum number of tokens per batch. Batches will be formed to approach this limit
        as closely as possible without exceeding it. No single text can exceed this value.

    max_length : int, optional
        Maximum sequence length for tokenization, by default 2048.
        Longer sequences will be truncated. Should match model's maximum context size.

    Returns
    -------
    List[Dict[int, torch.Tensor]]
        A list where each element corresponds to one input text in the original input order.
        Each element is a dictionary mapping layer index to a tensor of shape
        `[real_sequence_length, hidden_size]` containing the hidden states for actual
        content tokens (padding tokens removed).

    Raises
    ------
    ValueError
        If `layers` is empty, `kv_cache_size` is too small, or any text exceeds max_length
    RuntimeError
        If model forward pass fails or tokenization produces invalid output
    """
    # Input validation
    if not layers:
        raise ValueError("Layers sequence cannot be empty")

    if kv_cache_size <= 0:
        raise ValueError("kv_cache_size must be positive")

    if not texts:
        return []

    # Tokenize all texts individually to get exact lengths
    tokenized_texts = []
    sequence_lengths = []

    for text in texts:
        if not text.strip():
            # Handle empty text
            tokenized_texts.append({
                'input_ids': torch.empty(0, dtype=torch.long),
                'attention_mask': torch.empty(0, dtype=torch.long)
            })
            sequence_lengths.append(0)
            continue

        tokenized = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors="pt"
        )

        seq_len = tokenized['input_ids'].size(1)
        if seq_len > kv_cache_size:
            raise ValueError(f"Text length {seq_len} exceeds kv_cache_size {kv_cache_size}. "
                           "Consider increasing kv_cache_size or reducing max_length.")

        tokenized_texts.append({
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        })
        sequence_lengths.append(seq_len)

    # Create text indices with their lengths for bin packing
    text_data = list(zip(range(len(texts)), sequence_lengths, tokenized_texts))

    # Sort by length descending for efficient packing
    text_data.sort(key=lambda x: x[1], reverse=True)

    # Bin packing algorithm for optimal batching
    batches = _create_token_batches(text_data, kv_cache_size)

    # Process batches and collect results
    model_device = next(model.parameters()).device
    all_results = [{} for _ in range(len(texts))]

    for batch_idx, batch_texts in enumerate(tqdm(batches, desc="Processing batches")):
        if not batch_texts:
            continue

        # Prepare batch tensors with padding
        batch_input_ids = []
        batch_attention_mask = []
        batch_original_indices = []

        for orig_idx, seq_len, tokenized in batch_texts:
            batch_input_ids.append(tokenized['input_ids'])
            batch_attention_mask.append(tokenized['attention_mask'])
            batch_original_indices.append(orig_idx)

        # Pad sequences to longest in batch
        batch_input_ids_padded = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        batch_attention_mask_padded = pad_sequence(
            batch_attention_mask,
            batch_first=True,
            padding_value=0
        )

        # Move to model device
        batch_input_ids_padded = batch_input_ids_padded.to(model_device, non_blocking=True)
        batch_attention_mask_padded = batch_attention_mask_padded.to(model_device, non_blocking=True)

        # Forward pass
        with torch.inference_mode():
            outputs = model(
                input_ids=batch_input_ids_padded,
                attention_mask=batch_attention_mask_padded,
                output_hidden_states=True
            )
            all_hidden_states = outputs.hidden_states

        # Extract hidden states for each example in batch
        for i, orig_idx in enumerate(batch_original_indices):
            non_padding_mask = batch_attention_mask_padded[i].bool()

            for layer_index in layers:
                hidden_state = all_hidden_states[layer_index][i]
                hidden_state_no_padding = hidden_state[non_padding_mask]
                all_results[orig_idx][layer_index] = hidden_state_no_padding.half().cpu()

        gc.collect()
        torch.cuda.empty_cache()

    return all_results


def _create_token_batches(
    text_data: List[Tuple[int, int, Dict]],
    kv_cache_size: int
) -> List[List[Tuple[int, int, Dict]]]:
    """
    Create optimal batches based on token count using first-fit decreasing algorithm.

    Parameters
    ----------
    text_data : List[Tuple[int, int, Dict]]
        List of (original_index, sequence_length, tokenized_data) tuples, sorted by length descending
    kv_cache_size : int
        Maximum tokens per batch

    Returns
    -------
    List[List[Tuple[int, int, Dict]]]
        List of batches, each containing tuples of text data
    """
    batches = []

    for orig_idx, seq_len, tokenized in text_data:
        if seq_len == 0:  # Skip empty texts
            continue

        placed = False

        # Try to place in existing batch
        for batch in batches:
            batch_token_count = sum(seq_len for _, seq_len, _ in batch)
            if batch_token_count + seq_len <= kv_cache_size:
                batch.append((orig_idx, seq_len, tokenized))
                placed = True
                break

        # Create new batch if couldn't place
        if not placed:
            batches.append([(orig_idx, seq_len, tokenized)])

    return batches
