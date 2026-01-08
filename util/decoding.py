"""
Autoregressive decoding utilities for sequence generation.
Implements greedy decoding for transformer-based translation.
"""
import torch


def greedy_decode(model, src, max_len, sos_idx, eos_idx, pad_idx, device):
    """
    Generate a translation using greedy decoding (autoregressive generation).
    
    Args:
        model: Transformer model
        src: Source tensor of shape (1, src_len) - single sequence
        max_len: Maximum length of generated sequence
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index  
        pad_idx: Padding token index
        device: Device to run on
        
    Returns:
        Generated token indices tensor of shape (1, generated_len)
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source sequence
        src_mask = model.make_src_mask(src)
        enc_src = model.encoder(src, src_mask)
        
        # Initialize target with <sos> token
        trg = torch.tensor([[sos_idx]], device=device)
        
        for _ in range(max_len - 1):
            trg_mask = model.make_trg_mask(trg)
            output = model.decoder(trg, enc_src, trg_mask, src_mask)
            
            # Greedy selection: take token with highest probability
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            trg = torch.cat([trg, next_token], dim=1)
            
            # Stop if <eos> token is generated
            if next_token.item() == eos_idx:
                break
        
        return trg


def batch_greedy_decode(model, src_batch, max_len, sos_idx, eos_idx, pad_idx, device, verbose=False):
    """
    Generate translations for a batch using greedy decoding.
    Processes each sequence individually to handle variable output lengths.
    
    Args:
        model: Transformer model
        src_batch: Source tensor of shape (batch_size, src_len)
        max_len: Maximum length of generated sequences
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index
        pad_idx: Padding token index
        device: Device to run on
        verbose: If True, print progress every 20 sequences
        
    Returns:
        List of generated token indices tensors (one per batch element)
    """
    model.eval()
    batch_size = src_batch.shape[0]
    generated = []
    
    with torch.no_grad():
        for i in range(batch_size):
            if verbose and i > 0 and i % 20 == 0:
                print(f'      Generated {i}/{batch_size} sequences in this batch...')
            src = src_batch[i:i+1]  # Keep batch dimension
            output = greedy_decode(model, src, max_len, sos_idx, eos_idx, pad_idx, device)
            generated.append(output.squeeze(0))  # Remove batch dimension
    
    return generated


def idx_to_text(indices, vocab_transform, skip_special=True):
    """
    Convert token indices to text string.
    
    Args:
        indices: Tensor or list of token indices
        vocab_transform: Vocabulary object with itos mapping
        skip_special: If True, skip tokens containing '<' (special tokens)
        
    Returns:
        String of space-separated words
    """
    words = []
    
    if not hasattr(vocab_transform, 'itos') or vocab_transform.itos is None:
        raise AttributeError(
            "VocabTransform does not have 'itos' (index-to-token mapping) built. "
            "Did you forget to call build_vocab()?"
        )
    
    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(vocab_transform.itos):
            word = vocab_transform.itos[idx]
        else:
            word = '<unk>'
        
        if skip_special and '<' in word:
            continue
        words.append(word)
    
    return " ".join(words)
