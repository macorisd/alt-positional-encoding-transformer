#!/usr/bin/env python3
"""
Script to run 10-fold Cross Validation with different periodic functions
for Positional Encoding in the Transformer.

Usage:
    python test.py            # Start new cross-validation
    python test.py --resume   # Resume from most recent results directory
"""

import os
import sys
import math
import time
import shutil
import argparse
import glob
from datetime import datetime
import torch
from torch import nn, optim
from torch.optim import Adam

from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time


def get_most_recent_results_dir(base_path='/home/ubuntu/Desktop/GitHub/transformer-translation/results'):
    """Find the most recent results_[timestamp] directory"""
    if not os.path.exists(base_path):
        return None
    
    result_dirs = glob.glob(os.path.join(base_path, 'results_*'))
    if not result_dirs:
        return None
    
    # Sort by modification time, most recent first
    result_dirs.sort(key=os.path.getmtime, reverse=True)
    return result_dirs[0]


def find_resume_point(results_dir, periodic_functions, k_folds=10):
    """
    Find where to resume training.
    Returns: (function_to_resume, fold_to_resume) or (None, None) if complete
    """
    for periodic_func in periodic_functions:
        func_dir = os.path.join(results_dir, periodic_func)
        
        if not os.path.exists(func_dir):
            # This function hasn't been started yet
            return periodic_func, 0
        
        for fold_idx in range(k_folds):
            fold_dir = os.path.join(func_dir, f'{periodic_func}_{fold_idx + 1}')
            model_path = os.path.join(fold_dir, 'model_best.pt')
            
            if not os.path.exists(model_path):
                # This fold is incomplete, delete it and start from here
                if os.path.exists(fold_dir):
                    print(f"Removing incomplete fold: {fold_dir}")
                    shutil.rmtree(fold_dir)
                return periodic_func, fold_idx
    
    # All complete
    return None, None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def prepare_data_loaders():
    """Prepare data loaders for cross validation"""
    tokenizer = Tokenizer()
    loader = DataLoader(ext=('en', 'de'),
                       tokenize_en=tokenizer.tokenize_en,
                       tokenize_de=tokenizer.tokenize_de,
                       init_token='<sos>',
                       eos_token='<eos>')
    
    train_data, valid_data, test_data = loader.make_dataset()
    loader.build_vocab(train_data=train_data, min_freq=2)
    
    return loader, train_data, valid_data, test_data


def create_k_fold_splits(train_data, k=10):
    """Split training data into k folds"""
    import random
    random.seed(42)  # For reproducibility
    
    data_copy = list(train_data)
    random.shuffle(data_copy)
    
    fold_size = len(data_copy) // k
    folds = []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k - 1 else len(data_copy)
        folds.append(data_copy[start_idx:end_idx])
    
    return folds


def get_fold_data(folds, fold_idx):
    """Get training and validation data for a specific fold"""
    val_data = folds[fold_idx]
    train_data = []
    for i, fold in enumerate(folds):
        if i != fold_idx:
            train_data.extend(fold)
    return train_data, val_data


def train_epoch(model, iterator, optimizer, criterion, clip):
    """Train one epoch"""
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, trg = batch

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        
        if i % 10 == 0:
            print(f'  Step {i}/{len(iterator)} ({round((i / len(iterator)) * 100, 1)}%) - Loss: {loss.item():.4f}')
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, loader):
    """Evaluate the model"""
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_flat)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(trg.shape[0]):
                try:
                    trg_words = idx_to_word(trg[j], loader.target)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            if total_bleu:
                avg_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(avg_bleu)

    batch_bleu_score = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
    return epoch_loss / len(iterator), batch_bleu_score


def run_training(model, train_iter, valid_iter, loader, output_dir, total_epochs, device):
    """Run complete training"""
    optimizer = Adam(params=model.parameters(),
                    lr=init_lr,
                    weight_decay=weight_decay,
                    eps=adam_eps)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     factor=factor,
                                                     patience=patience)
    
    criterion = nn.CrossEntropyLoss(ignore_index=loader.source.vocab['<pad>'])
    
    train_losses, test_losses, bleus = [], [], []
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(total_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion, loader)
        
        end_time = time.time()
        
        if epoch > warmup:
            scheduler.step(valid_loss)
        
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Track best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_state = model.state_dict().copy()
        
        # Save results after each epoch
        with open(os.path.join(output_dir, 'train_loss.txt'), 'w') as f:
            f.write(str(train_losses))
        
        with open(os.path.join(output_dir, 'test_loss.txt'), 'w') as f:
            f.write(str(test_losses))
        
        with open(os.path.join(output_dir, 'bleu.txt'), 'w') as f:
            f.write(str(bleus))
        
        print(f'Epoch {epoch+1}/{total_epochs} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'  Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'  Val Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'  BLEU Score: {bleu:.3f}')
    
    # Save best model at the end of training
    if best_model_state is not None:
        model_path = os.path.join(output_dir, 'model_best.pt')
        torch.save(best_model_state, model_path)
        print(f'Best model saved with validation loss: {best_loss:.4f}')
    
    return train_losses, test_losses, bleus


def run_cross_validation(periodic_func, loader, folds, base_output_dir, total_epochs=1000, start_fold=0):
    """Run cross validation for a specific periodic function"""
    func_dir = os.path.join(base_output_dir, periodic_func)
    os.makedirs(func_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CROSS VALIDATION: {periodic_func}")
    if start_fold > 0:
        print(f"RESUMING FROM FOLD {start_fold + 1}")
    print(f"{'='*80}\n")
    
    # Load existing results if resuming
    all_results = []
    if start_fold > 0:
        # Load stats from completed folds
        for prev_fold_idx in range(start_fold):
            fold_dir = os.path.join(func_dir, f'{periodic_func}_{prev_fold_idx + 1}')
            if os.path.exists(fold_dir):
                # Try to reconstruct stats from saved files
                try:
                    with open(os.path.join(fold_dir, 'train_loss.txt'), 'r') as f:
                        train_losses = eval(f.read())
                    with open(os.path.join(fold_dir, 'test_loss.txt'), 'r') as f:
                        test_losses = eval(f.read())
                    with open(os.path.join(fold_dir, 'bleu.txt'), 'r') as f:
                        bleus = eval(f.read())
                    
                    fold_stats = {
                        'fold': prev_fold_idx + 1,
                        'final_train_loss': train_losses[-1],
                        'final_val_loss': test_losses[-1],
                        'final_bleu': bleus[-1],
                        'best_val_loss': min(test_losses),
                        'best_bleu': max(bleus)
                    }
                    all_results.append(fold_stats)
                    print(f"Loaded existing results for Fold {prev_fold_idx + 1}")
                except:
                    print(f"Warning: Could not load results for Fold {prev_fold_idx + 1}")
    
    for fold_idx in range(start_fold, len(folds)):
        print(f"\n{'-'*80}")
        print(f"Fold {fold_idx + 1}/{len(folds)}")
        print(f"{'-'*80}")
        
        # Create directory for this fold
        fold_dir = os.path.join(func_dir, f'{periodic_func}_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Prepare data for this fold
        train_data, val_data = get_fold_data(folds, fold_idx)
        
        # Create iterators
        train_iter, val_iter, _ = loader.make_iter(train_data, val_data, val_data,
                                                    batch_size=batch_size,
                                                    device=device)
        
        # Create model
        model = Transformer(src_pad_idx=loader.source.vocab['<pad>'],
                          trg_pad_idx=loader.target.vocab['<pad>'],
                          trg_sos_idx=loader.target.vocab['<sos>'],
                          d_model=d_model,
                          enc_voc_size=len(loader.source.vocab),
                          dec_voc_size=len(loader.target.vocab),
                          max_len=max_len,
                          ffn_hidden=ffn_hidden,
                          n_head=n_heads,
                          n_layers=n_layers,
                          drop_prob=drop_prob,
                          device=device,
                          periodic_func=periodic_func).to(device)
        
        model.apply(initialize_weights)
        
        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Periodic function: {periodic_func}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}\n")
        
        # Train
        train_losses, test_losses, bleus = run_training(
            model, train_iter, val_iter, loader, fold_dir, total_epochs, device
        )
        
        # Save fold statistics
        fold_stats = {
            'fold': fold_idx + 1,
            'final_train_loss': train_losses[-1],
            'final_val_loss': test_losses[-1],
            'final_bleu': bleus[-1],
            'best_val_loss': min(test_losses),
            'best_bleu': max(bleus)
        }
        all_results.append(fold_stats)
        
        print(f"\nFold {fold_idx + 1} completed:")
        print(f"  Final Train Loss: {fold_stats['final_train_loss']:.4f}")
        print(f"  Final Val Loss: {fold_stats['final_val_loss']:.4f}")
        print(f"  Final BLEU: {fold_stats['final_bleu']:.4f}")
        print(f"  Best Val Loss: {fold_stats['best_val_loss']:.4f}")
        print(f"  Best BLEU: {fold_stats['best_bleu']:.4f}")
    
    # Save results summary
    summary_path = os.path.join(func_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Cross Validation Summary: {periodic_func}\n")
        f.write(f"{'='*60}\n\n")
        
        for stats in all_results:
            f.write(f"Fold {stats['fold']}:\n")
            f.write(f"  Final Train Loss: {stats['final_train_loss']:.4f}\n")
            f.write(f"  Final Val Loss: {stats['final_val_loss']:.4f}\n")
            f.write(f"  Final BLEU: {stats['final_bleu']:.4f}\n")
            f.write(f"  Best Val Loss: {stats['best_val_loss']:.4f}\n")
            f.write(f"  Best BLEU: {stats['best_bleu']:.4f}\n\n")
        
        # Averages
        avg_train_loss = sum(s['final_train_loss'] for s in all_results) / len(all_results)
        avg_val_loss = sum(s['final_val_loss'] for s in all_results) / len(all_results)
        avg_bleu = sum(s['final_bleu'] for s in all_results) / len(all_results)
        avg_best_bleu = sum(s['best_bleu'] for s in all_results) / len(all_results)
        
        f.write(f"\nAVERAGE ACROSS ALL FOLDS:\n")
        f.write(f"  Avg Final Train Loss: {avg_train_loss:.4f}\n")
        f.write(f"  Avg Final Val Loss: {avg_val_loss:.4f}\n")
        f.write(f"  Avg Final BLEU: {avg_bleu:.4f}\n")
        f.write(f"  Avg Best BLEU: {avg_best_bleu:.4f}\n")
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {periodic_func}")
    print(f"Average Final BLEU: {avg_bleu:.4f}")
    print(f"Average Best BLEU: {avg_best_bleu:.4f}")
    print(f"{'='*80}\n")
    
    return all_results


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='10-fold Cross Validation for Transformer')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from most recent results directory')
    args = parser.parse_args()
    
    # Configuration
    periodic_functions = ['sinusoid', 'triangular', 'square', 'sawtooth']
    k_folds = 10
    epochs_per_fold = 1000
    
    # Determine base directory and resume point
    if args.resume:
        base_dir = get_most_recent_results_dir()
        if base_dir is None:
            print("No previous results directory found. Starting new training.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = f'/home/ubuntu/Desktop/GitHub/transformer-translation/results/results_{timestamp}'
            os.makedirs(base_dir, exist_ok=True)
            resume_function = None
            resume_fold = 0
        else:
            print(f"Found previous results: {base_dir}")
            resume_function, resume_fold = find_resume_point(base_dir, periodic_functions, k_folds)
            
            if resume_function is None:
                print("All training already completed!")
                return
            
            print(f"Resuming from: {resume_function}, fold {resume_fold + 1}")
    else:
        # Create new base directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f'/home/ubuntu/Desktop/GitHub/transformer-translation/results/results_{timestamp}'
        os.makedirs(base_dir, exist_ok=True)
        resume_function = None
        resume_fold = 0
    
    print(f"\n{'#'*80}")
    print(f"# 10-FOLD CROSS VALIDATION - POSITIONAL ENCODING COMPARISON")
    print(f"# Timestamp: {os.path.basename(base_dir)}")
    print(f"# Output directory: {base_dir}")
    print(f"# K-folds: {k_folds}")
    print(f"# Epochs per fold: {epochs_per_fold}")
    print(f"# Periodic functions: {', '.join(periodic_functions)}")
    if args.resume and resume_function:
        print(f"# RESUMING: {resume_function} from fold {resume_fold + 1}")
    print(f"{'#'*80}\n")
    
    # Prepare data
    print("Loading and preparing data...")
    loader, train_data, valid_data, test_data = prepare_data_loaders()
    
    print(f"Total training samples: {len(train_data)}")
    print(f"Creating {k_folds} folds...")
    folds = create_k_fold_splits(train_data, k=k_folds)
    
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: {len(fold)} samples")
    
    # Run cross validation for each periodic function
    all_function_results = {}
    
    # Determine which functions to run
    if resume_function:
        func_start_idx = periodic_functions.index(resume_function)
        functions_to_run = periodic_functions[func_start_idx:]
    else:
        functions_to_run = periodic_functions
    
    for idx, periodic_func in enumerate(functions_to_run):
        # Only use resume_fold for the first function when resuming
        start_fold = resume_fold if idx == 0 and resume_function == periodic_func else 0
        
        results = run_cross_validation(
            periodic_func, loader, folds, base_dir, epochs_per_fold, start_fold=start_fold
        )
        all_function_results[periodic_func] = results
    
    # Create global summary
    global_summary_path = os.path.join(base_dir, 'global_summary.txt')
    with open(global_summary_path, 'w') as f:
        f.write("GLOBAL SUMMARY - ALL PERIODIC FUNCTIONS\n")
        f.write(f"{'='*80}\n\n")
        
        for func, results in all_function_results.items():
            avg_bleu = sum(s['final_bleu'] for s in results) / len(results)
            avg_best_bleu = sum(s['best_bleu'] for s in results) / len(results)
            
            f.write(f"{func}:\n")
            f.write(f"  Average Final BLEU: {avg_bleu:.4f}\n")
            f.write(f"  Average Best BLEU: {avg_best_bleu:.4f}\n\n")
    
    print(f"\n{'#'*80}")
    print(f"# ALL CROSS VALIDATIONS COMPLETED")
    print(f"# Results saved in: {base_dir}")
    print(f"# Check global_summary.txt for comparison")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
