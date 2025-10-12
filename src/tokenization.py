"""
🔬 Advanced Text Tokenization Module with Parallel Processing Support

This module provides high-performance tokenization functionality for transformer-based language models.
It is designed for researchers and practitioners who need efficient tokenization of large text corpora
with support for parallel processing and detailed progress tracking.

🎯 Key Features
- Parallel Tokenization: Multi-process support for accelerated processing of large datasets
- Flexible Progress Tracking: Configurable tqdm progress bars with token statistics
- Memory Efficient: Batch processing with configurable chunk sizes
- Error Resilient: Robust error handling with configurable error responses
- Order Preservation: Guaranteed output order matches input order

🚀 Use Cases
- Large-scale text preprocessing for model training
- Feature extraction pipelines
- Linguistic analysis and corpus studies
- Data preparation for transformer models

💡 Key Concepts

1. Tokenizer Callable
   Any function that takes a string and returns a list of tokens (strings)
2. Parallel Processing
   Utilizes multiple CPU cores for faster tokenization of large datasets
3. Progress Tracking
   Real-time monitoring of processing speed, token counts, and time estimates

📝 Usage Pattern

```python
from transformers import AutoTokenizer
from tokenization_module import tokenize

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize texts with parallel processing
texts = ["Hello world!", "Another example text"]
results = tokenize(
    texts=texts,
    tokenizer_callable=tokenizer,
    num_workers=4,
    batch_size=1000,
    show_progress=True,
    desc="Tokenizing corpus"
)
```
"""


import time
from typing import Any, List, Callable, Literal, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm


def tokenize(
    texts: Union[List[str], pd.Series],
    tokenizer_callable: Any,
    num_workers: int = 1,
    batch_size: int = 1000,
    show_progress: bool = True,
    desc: Optional[str] = None,
    on_error: Literal['keep', 'skip', 'empty'] = 'keep'
) -> List[List[str]]:
    """
    Tokenize a collection of texts into tokens with parallel processing support.

    This function processes input texts using the provided tokenizer function and returns
    a list of tokenized texts. Supports parallel processing for large datasets and provides
    detailed progress tracking with token statistics.

    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Input texts to tokenize. Can be a list of strings or pandas Series.

    tokenizer_callable : Callable[[str], List[str]]
        Class with funtions `tokenize` and `convert_tokens_to_string` that takes a string
        and returns a list of tokens (strings). Typically a tokenizer from Hugging Face transformers.

    num_workers : int, optional
        Number of parallel workers to use. Default is 1 (sequential processing).
        Set to 0 or 1 for sequential processing, >1 for parallel processing.

    batch_size : int, optional
        Number of texts to process in each batch for parallel processing.
        Default is 1000. Smaller batches reduce memory usage, larger batches
        improve parallelization efficiency.

    show_progress : bool, optional
        Whether to display a progress bar with statistics. Default is True.

    desc : str, optional
        Description to display in the progress bar. If None, a default description
        is used based on the number of workers.

    on_error : Literal['keep', 'skip', 'empty'], optional
        How to handle tokenization errors:
        - 'keep': Keep original text as single token
        - 'skip': Skip problematic texts (returns empty list for that position)
        - 'empty': Return empty token list for problematic texts
        Default is 'keep'.

    Returns
    -------
    List[List[str]]
        A list where each element corresponds to one input text in the original order.
        Each element is a list of tokens (strings) for that text.

    Raises
    ------
    ValueError
        If `num_workers` is negative, `batch_size` is not positive, or `on_error`
        has an invalid value.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> texts = ["Hello world!", "This is a test."]
    >>> results = tokenize_texts(texts, tokenizer, num_workers=2)
    >>> print(results)
    [['hello', 'world', '!'], ['this', 'is', 'a', 'test', '.']]
    """
    # Input validation
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if on_error not in ['keep', 'skip', 'empty']:
        raise ValueError("on_error must be one of: 'keep', 'skip', 'empty'")

    if not texts: # type: ignore
        return []

    # Convert pandas Series to list if necessary
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    # Default progress bar description
    if desc is None:
        if num_workers <= 1:
            desc = "Tokenizing"
        else:
            desc = f"Tokenizing ({num_workers} workers)"

    # Sequential processing
    if num_workers <= 1:
        results = []
        total_tokens = 0

        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(total=len(texts), desc=desc)

        start_time = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = _process_batch(batch, tokenizer_callable.tokenize, on_error)
            batch_results = [[tokenizer_callable.convert_tokens_to_string([token]) for token in text] for text in batch_results]
            results.extend(batch_results)

            # Update statistics for progress bar
            if show_progress:
                batch_tokens = sum(len(tokens) for tokens in batch_results)
                total_tokens += batch_tokens
                pbar.update(len(batch))

                # Calculate and display statistics
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                texts_per_sec = len(results) / elapsed if elapsed > 0 else 0

                pbar.set_postfix({
                    'texts/sec': f'{texts_per_sec:.1f}',
                    'tokens/sec': f'{tokens_per_sec:.1f}',
                    'total_tokens': total_tokens
                })

        if show_progress:
            pbar.close()

    # Parallel processing
    else:
        results = [None] * len(texts)  # Pre-allocate results list to maintain order
        total_tokens = 0
        completed_indices = 0

        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(total=len(texts), desc=desc)

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches for processing
            future_to_batch = {}
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_indices = list(range(i, min(i + batch_size, len(texts))))
                future = executor.submit(_process_batch, batch, tokenizer_callable.tokenize, on_error)
                future_to_batch[future] = batch_indices

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_indices = future_to_batch[future]
                try:
                    batch_results = future.result()

                    batch_results = [[tokenizer_callable.convert_tokens_to_string([token]) for token in text] for text in batch_results]

                    # Store results in correct positions
                    for idx, result in zip(batch_indices, batch_results):
                        results[idx] = result

                    # Update progress bar
                    if show_progress:
                        completed_indices += len(batch_indices)
                        batch_tokens = sum(len(tokens) for tokens in batch_results)
                        total_tokens += batch_tokens
                        pbar.update(len(batch_indices))

                        # Calculate and display statistics
                        elapsed = time.time() - start_time
                        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                        texts_per_sec = completed_indices / elapsed if elapsed > 0 else 0

                        pbar.set_postfix({
                            'texts/sec': f'{texts_per_sec:.1f}',
                            'tokens/sec': f'{tokens_per_sec:.1f}',
                            'total_tokens': total_tokens
                        })

                except Exception as e:
                    # Handle batch processing errors
                    if show_progress:
                        pbar.write(f"Error processing batch: {e}")
                    # Fill batch positions according to error strategy
                    for idx in batch_indices:
                        if on_error == 'keep':
                            results[idx] = [texts[idx]]
                        else:
                            results[idx] = []

        if show_progress:
            pbar.close()

    return results # type: ignore


def _process_batch(batch, tokenizer_callable, on_error) -> List[List[str]]:
    """Process a single batch of texts."""
    results = []
    for text in batch:
        try:
            results.append(tokenizer_callable(text))
        except Exception:
            if on_error == 'keep':
                results.append([text])
            else:
                results.append([])
    return results


# Convenience function for common tokenization patterns
def create_tokenizer_from_huggingface(
    model_name: str,
    **tokenizer_kwargs
) -> Callable[[str], List[str]]:
    """
    Create a tokenizer callable from a Hugging Face model name.

    Parameters
    ----------
    model_name : str
        Name of the pre-trained model on Hugging Face Model Hub.

    **tokenizer_kwargs
        Additional arguments to pass to the tokenizer.

    Returns
    -------
    Callable[[str], List[str]]
        A tokenizer function that can be used with tokenize_texts.

    Examples
    --------
    >>> tokenizer = create_tokenizer_from_huggingface("bert-base-uncased")
    >>> tokens = tokenizer("Hello world!")
    >>> print(tokens)
    ['hello', 'world', '!']
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        return tokenizer
    except ImportError:
        raise ImportError(
            "transformers package is required for Hugging Face tokenizers. "
            "Install with: pip install transformers"
        )
