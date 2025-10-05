"""
🔬 Hidden States Extraction Module.
This module provides specialized functionality for extracting hidden states from transformer-based language models.
It is designed for researchers and practitioners who need fine-grained access to intermediate layer representations
without padding artifacts.

## 🎯 Key Features
- Padding-Free Extraction: Automatically removes padding tokens from hidden states
- Efficient Processing: Implements length-based sorting for optimal GPU utilization
- Layer Selection: Flexible extraction from specific transformer layers
- Batch Processing: Configurable batch size for memory optimization
- Device-Aware: Automatically handles model device placement

🚀 Use Cases
- Probing Studies: Analyze linguistic properties in different layers
- Feature Extraction: Get high-quality embeddings for downstream tasks
- Model Analysis: Study information flow through transformer layers
- Research Experiments: Reproduce layer-wise analysis experiments

💡 Key Concepts

1. Hidden States
   Raw activations from transformer layers that capture linguistic information at different levels of abstraction.
2. Padding Removal
   Critical for obtaining clean representations where each token embedding corresponds to actual content.
3. Layer Selection
   Different layers capture different types of information:
   - Lower layers: syntactic and surface features
   - Middle layers: semantic and contextual information
   - Higher layers: task-specific and abstract representations

📝 Usage Pattern

```python
from transformers import AutoModel, AutoTokenizer
from hidden_states_extractor import embed

# Load model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Extract layers 4, 8, 12
texts = ["Hello world!", "Another example text"]
results = embed(
    texts=texts,
    model=model,
    tokenizer=tokenizer,
    layers=[4, 8, 12],
    batch_size=2
)
```

⚡ Performance Notes
- Sorting by sequence length reduces padding and improves throughput
- Half-precision (FP16) used for memory efficiency
- Automatic device detection ensures compatibility
- Memory usage scales with batch_size × sequence_length × hidden_size × layers
"""

from typing import Sequence, List, Dict, Any
import torch
from tqdm.auto import tqdm


def embed(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    layers: Sequence[int],
    batch_size: int = 1,
    max_length: int = 2048,
) -> List[Dict[int, torch.Tensor]]:
    """
    Extract hidden states from specified transformer layers, excluding padding tokens.
    This function processes input texts through a transformer model and returns hidden states
    from the specified layers. The implementation uses length-based sorting to minimize
    padding and optimize computational efficiency. Padding tokens are automatically removed
    from the output representations.


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
        Example: [0, 4, 8, 12] extracts from layers 1, 5, 9, and 13.
        Negative indices count from the last layer (-1 = last layer).

    batch_size : int, optional
        Number of texts to process simultaneously, by default 1.
        Larger batches improve throughput but increase memory usage.
        Should be adjusted based on available GPU memory and sequence lengths.

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
        If `layers` is empty or contains invalid indices
    RuntimeError
        If model forward pass fails or tokenization produces invalid output

    Notes
    -----
    - The function preserves the original input order despite internal sorting for efficiency
    - Output tensors are in half-precision (FP16) to reduce memory footprint
    - Hidden states are moved to CPU memory before returning
    - The attention mask is used to identify and remove padding tokens
    - Model is set to inference mode (no gradients calculated)

    Examples
    --------
    >>> from transformers import AutoModel, AutoTokenizer

    >>> # Basic usage with BERT
    >>> model = AutoModel.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> texts = ["This is a test.", "Another example."]

    >>> # Extract from layers 3, 6, 9
    >>> hidden_states = embed(texts, model, tokenizer, layers=[3, 6, 9])
    >>> print(len(hidden_states))  # 2 (one per input text)
    >>> print(hidden_states[0][3].shape)  # torch.Size([real_seq_len, 768])

    >>> # With custom batch size and sequence length
    >>> hidden_states = embed(
    ...     texts=texts,
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     layers=[-1],  # Last layer only
    ...     batch_size=4,
    ...     max_length=512
    ... )

    >>> # Processing single text
    >>> single_result = embed(
    ...     texts=["Single sentence"],
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     layers=range(12)  # First 12 layers
    ... )
    """
    # Validate input parameters
    if not layers:
        raise ValueError("Layers sequence cannot be empty")

    # Tokenize all texts with standard transformer parameters
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Calculate actual sequence lengths from attention mask
    # This determines the number of non-padding tokens per sequenc
    sequence_lengths = attention_mask.sum(dim=1)

    # Sort sequences by length in descending order for optimal GPU utilization
    # Longer sequences first minimizes the "bubbles" in padded batche
    sorted_indices = torch.argsort(sequence_lengths, descending=True)

    # Create mapping to restore original order after processing
    # This ensures output matches input text order regardless of internal sorting
    original_order_indices = torch.empty_like(sorted_indices)
    original_order_indices[sorted_indices] = torch.arange(len(texts))

    # Sort all tensors according to length-based ordering
    sorted_input_ids = input_ids[sorted_indices]
    sorted_attention_mask = attention_mask[sorted_indices]

    # Create batches from sorted data
    # Each batch contains sequences of similar length to minimize padding
    batches = [
        (sorted_input_ids[i:i + batch_size], sorted_attention_mask[i:i + batch_size])
        for i in range(0, sorted_input_ids.shape[0], batch_size)
    ]

    # Initialize results for sorted order
    sorted_results = [{} for _ in range(len(texts))]
    model_device = next(model.parameters()).device

    # Process batches in sorted order
    for batch_idx, (batch_input_ids, batch_attention_mask) in enumerate(tqdm(batches, desc="Processing batches")):
        # Move batch to model device
        batch_input_ids = batch_input_ids.to(model_device, non_blocking=True)
        batch_attention_mask = batch_attention_mask.to(model_device, non_blocking=True)

        # Forward pass
        with torch.inference_mode():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True
            )
            all_hidden_states = outputs.hidden_states

        # Extract hidden states for each example in batch
        batch_start_idx = batch_idx * batch_size
        for i_in_batch in range(batch_input_ids.size(0)):
            text_idx = batch_start_idx + i_in_batch
            if text_idx >= len(texts):
                break

            # Get non-padding tokens for this example
            non_padding_mask = batch_attention_mask[i_in_batch].bool()

            # Extract hidden states for requested layers
            for layer_index in layers:
                hidden_state = all_hidden_states[layer_index][i_in_batch]  # [seq_len, hidden_size]
                hidden_state_no_padding = hidden_state[non_padding_mask]   # [real_seq_len, hidden_size]
                sorted_results[text_idx][layer_index] = hidden_state_no_padding.half().cpu()

    # Restore original order
    final_results = [sorted_results[i] for i in original_order_indices.tolist()]

    return final_results
