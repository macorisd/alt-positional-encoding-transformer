"""
BLEU score calculation using sacreBLEU.
Provides standardized, reproducible BLEU scores compatible with academic benchmarks.
"""
from sacrebleu import corpus_bleu, BLEU

from util.decoding import idx_to_text, batch_greedy_decode


def compute_bleu(hypotheses, references, lowercase=False):
    """
    Compute corpus-level BLEU score using sacreBLEU.
    
    Args:
        hypotheses: List of hypothesis strings (generated translations)
        references: List of reference strings (ground truth translations)
        lowercase: If True, compute case-insensitive BLEU
        
    Returns:
        BLEU score (0-100 scale)
    """
    # sacreBLEU expects references as list of lists (supports multiple references per hypothesis)
    refs_wrapped = [[ref] for ref in references]
    
    # Transpose for sacreBLEU format: list of reference lists, one per reference set
    refs_transposed = list(zip(*refs_wrapped))
    refs_transposed = [list(r) for r in refs_transposed]
    
    bleu = corpus_bleu(hypotheses, refs_transposed, lowercase=lowercase)
    return bleu.score


def compute_bleu_detailed(hypotheses, references, lowercase=False):
    """
    Compute BLEU score with detailed breakdown.
    
    Args:
        hypotheses: List of hypothesis strings
        references: List of reference strings
        lowercase: If True, compute case-insensitive BLEU
        
    Returns:
        dict with 'score', 'signature', 'precisions', 'bp' (brevity penalty)
    """
    refs_wrapped = [[ref] for ref in references]
    refs_transposed = list(zip(*refs_wrapped))
    refs_transposed = [list(r) for r in refs_transposed]
    
    bleu = corpus_bleu(hypotheses, refs_transposed, lowercase=lowercase)
    
    return {
        'score': bleu.score,
        'signature': bleu.format(),
        'precisions': bleu.precisions,
        'bp': bleu.bp
    }


def evaluate_bleu(model, iterator, vocab_target, max_len, sos_idx, eos_idx, pad_idx, device):
    """
    Evaluate model using autoregressive generation and sacreBLEU.
    
    Args:
        model: Transformer model
        iterator: Data iterator yielding (src, trg) batches
        vocab_target: Target vocabulary with itos mapping
        max_len: Maximum generation length
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index
        pad_idx: Padding token index
        device: Device to run on
        
    Returns:
        BLEU score (0-100 scale)
    """
    model.eval()
    
    all_hypotheses = []
    all_references = []
    
    for batch_idx, batch in enumerate(iterator):
        print(f'    Processing batch {batch_idx + 1} for BLEU calculation...')
        
        src, trg = batch
        
        # Generate translations autoregressively
        generated = batch_greedy_decode(
            model, src, max_len, sos_idx, eos_idx, pad_idx, device, verbose=True
        )
        
        # Convert to text
        for i, gen in enumerate(generated):
            hyp = idx_to_text(gen, vocab_target)
            ref = idx_to_text(trg[i], vocab_target)
            
            all_hypotheses.append(hyp)
            all_references.append(ref)
    
    return compute_bleu(all_hypotheses, all_references)


# Backward compatibility alias
def idx_to_word(x, vocab_transform):
    """Alias for idx_to_text for backward compatibility."""
    return idx_to_text(x, vocab_transform)
