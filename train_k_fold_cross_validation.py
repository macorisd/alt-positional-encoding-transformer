#!/usr/bin/env python3
"""
Script to run 10-fold Cross Validation with different periodic functions
for Positional Encoding in the Transformer.

Usage:
    python train_k_fold_cross_validation.py
        Start new cross-validation with all functions (uses device from conf.py)
    
    python train_k_fold_cross_validation.py --resume
        Resume from most recent results directory
    
    python train_k_fold_cross_validation.py --functions sinusoid,triangular --device cuda:0
        Train only specific functions on specific GPU
    
    python train_k_fold_cross_validation.py --functions square,sawtooth --device cuda:1
        Train different functions on different GPU (for parallel execution)

Arguments:
    --resume            Resume from most recent results directory
    --functions STR     Comma-separated periodic functions (default: sinusoid,triangular,square,sawtooth)
    --folds STR         Comma-separated fold numbers to train (1-based, e.g., 1,2,3,4,5). Default: all folds
    --device STR        Device override (e.g., cuda:0, cuda:1). Overrides conf.py
    --bleu-every INT    Calculate BLEU every N epochs (default: 25). Set to 1 for every epoch.
    --save-every INT    Save losses and BLEU to files every N epochs (default: 25). Values are retained in memory.
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
from util.bleu import evaluate_bleu
from util.epoch_timer import epoch_time


def get_most_recent_results_dir(base_path='/mnt/home/users/tic_163_uma/macorisd/GitHub/alt-positional-encoding-transformer/results'):
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


def filter_folds_to_run(results_dir, periodic_func, k_folds=10, user_specified_folds=None):
    """
    Get list of folds that need to be executed for a specific periodic function.
    Cleans up incomplete fold directories (those without model_best.pt).
    
    Args:
        results_dir: Base results directory
        periodic_func: Name of the periodic function
        k_folds: Total number of folds
        user_specified_folds: List of fold indices (0-based) that user wants to run, or None for all
    
    Returns: list of fold indices (0-based) that need to be executed, or None if all complete/skipped
    """
    func_dir = os.path.join(results_dir, periodic_func)
    
    # Determine which folds to check based on user specification
    folds_to_check = user_specified_folds if user_specified_folds is not None else list(range(k_folds))
    
    if not os.path.exists(func_dir):
        # This function hasn't been started yet - need to run specified folds
        return folds_to_check
    
    incomplete_folds = []
    for fold_idx in folds_to_check:
        fold_dir = os.path.join(func_dir, f'{periodic_func}_{fold_idx + 1}')
        model_path = os.path.join(fold_dir, 'model_best.pt')
        
        if not os.path.exists(model_path):
            # This fold is incomplete
            if os.path.exists(fold_dir):
                print(f"Removing incomplete fold: {fold_dir}")
                shutil.rmtree(fold_dir)
            incomplete_folds.append(fold_idx)
    
    # Return None if all complete, otherwise return list of incomplete folds
    return None if len(incomplete_folds) == 0 else incomplete_folds


def get_incomplete_folds(results_dir, periodic_func, k_folds=10):
    """
    Get list of folds that need to be executed for a specific periodic function.
    Cleans up incomplete fold directories (those without model_best.pt).
    Returns: list of fold indices (0-based) that need to be executed, or None if all complete
    """
    func_dir = os.path.join(results_dir, periodic_func)
    
    if not os.path.exists(func_dir):
        # This function hasn't been started yet - need to run all folds
        return list(range(k_folds))
    
    incomplete_folds = []
    for fold_idx in range(k_folds):
        fold_dir = os.path.join(func_dir, f'{periodic_func}_{fold_idx + 1}')
        model_path = os.path.join(fold_dir, 'model_best.pt')
        
        if not os.path.exists(model_path):
            # This fold is incomplete
            if os.path.exists(fold_dir):
                print(f"Removing incomplete fold: {fold_dir}")
                shutil.rmtree(fold_dir)
            incomplete_folds.append(fold_idx)
    
    # Return None if all complete, otherwise return list of incomplete folds
    return None if len(incomplete_folds) == 0 else incomplete_folds


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


def evaluate(model, iterator, criterion, loader, device, val_data, compute_bleu=True):
    """
    Evaluate model computing loss and optionally BLEU score.
    Uses autoregressive generation for BLEU calculation.
    
    Args:
        compute_bleu: If False, skip BLEU calculation and return None
        val_data: Validation data for creating BLEU iterator with smaller batch size
    """
    model.eval()
    epoch_loss = 0
    
    # Calculate loss with teacher forcing (uses training batch_size)
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_flat)
            epoch_loss += loss.item()
    
    # Calculate BLEU with autoregressive generation (uses smaller batch_size)
    bleu_score = None
    if compute_bleu:
        print(f'  Computing BLEU with autoregressive generation (batch_size={batch_size})...')
        
        # Create new iterator with batch size for BLEU
        _, bleu_iter, _ = loader.make_iter(val_data, val_data, val_data,
                                           batch_size=batch_size,
                                           device=device)
        
        sos_idx = loader.target.vocab['<sos>']
        eos_idx = loader.target.vocab['<eos>']
        pad_idx = loader.target.vocab['<pad>']
        
        bleu_score = evaluate_bleu(
            model=model,
            iterator=bleu_iter,
            vocab_target=loader.target,
            max_len=max_len,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            device=device
        )
    
    return epoch_loss / len(iterator), bleu_score


def run_training(model, train_iter, valid_iter, loader, val_data, output_dir, total_epochs, device, periodic_func, fold_info, bleu_every=25, save_every=25):
    """
    Run complete training.
    
    Args:
        periodic_func: Name of periodic function (e.g., 'sinusoid')
        fold_info: String like 'Fold 1/10'
        save_every: Save losses and BLEU to files every N epochs (default: 25)
    """
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
        
        # Calculate BLEU only every N epochs or on last epoch
        compute_bleu = (epoch + 1) % bleu_every == 0 or epoch == total_epochs - 1
        valid_loss, bleu = evaluate(model, valid_iter, criterion, loader, device, val_data, compute_bleu=compute_bleu)
        
        end_time = time.time()
        
        if epoch > warmup:
            scheduler.step(valid_loss)
        
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)  # Append None if not calculated, or the actual value
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Track best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_state = model.state_dict().copy()
        
        # Save results every save_every epochs or on last epoch
        if (epoch + 1) % save_every == 0 or epoch == total_epochs - 1:
            with open(os.path.join(output_dir, 'train_loss.txt'), 'w') as f:
                f.write(str(train_losses))
            
            with open(os.path.join(output_dir, 'test_loss.txt'), 'w') as f:
                f.write(str(test_losses))
            
            with open(os.path.join(output_dir, 'bleu.txt'), 'w') as f:
                f.write(str(bleus))
        
        print(f'[{periodic_func} | {fold_info}] Epoch {epoch+1}/{total_epochs} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'  Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'  Val Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f}')
        if bleu is not None:
            print(f'  BLEU Score: {bleu:.3f}')
        else:
            print(f'  BLEU Score: (skipped, calculated every {bleu_every} epochs)')
    
    # Save best model at the end of training
    if best_model_state is not None:
        model_path = os.path.join(output_dir, 'model_best.pt')
        torch.save(best_model_state, model_path)
        print(f'Best model saved with validation loss: {best_loss:.4f}')
    
    return train_losses, test_losses, bleus


def run_cross_validation(periodic_func, loader, folds, base_output_dir, total_epochs=1000, folds_to_run=None, bleu_every=25, save_every=25):
    """
    Run cross validation for a specific periodic function
    
    Args:
        folds_to_run: List of fold indices (0-based) to execute. If None, runs all folds.
        save_every: Save losses and BLEU to files every N epochs (default: 25)
    """
    func_dir = os.path.join(base_output_dir, periodic_func)
    os.makedirs(func_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CROSS VALIDATION: {periodic_func}")
    if folds_to_run is not None and len(folds_to_run) < len(folds):
        print(f"EXECUTING FOLDS: {[f+1 for f in folds_to_run]} (skipping completed folds)")
    print(f"{'='*80}\n")
    
    # Determine which folds to execute
    if folds_to_run is None:
        folds_to_run = list(range(len(folds)))
    
    # Load existing results from completed folds
    all_results = []
    for fold_idx in range(len(folds)):
        if fold_idx not in folds_to_run:
            # This fold is already complete, load its results
            fold_dir = os.path.join(func_dir, f'{periodic_func}_{fold_idx + 1}')
            if os.path.exists(fold_dir):
                try:
                    with open(os.path.join(fold_dir, 'train_loss.txt'), 'r') as f:
                        train_losses = eval(f.read())
                    with open(os.path.join(fold_dir, 'test_loss.txt'), 'r') as f:
                        test_losses = eval(f.read())
                    with open(os.path.join(fold_dir, 'bleu.txt'), 'r') as f:
                        bleus = eval(f.read())
                    
                    # Filter out None values from bleus before finding max
                    valid_bleus = [b for b in bleus if b is not None]
                    fold_stats = {
                        'fold': fold_idx + 1,
                        'final_train_loss': train_losses[-1],
                        'final_val_loss': test_losses[-1],
                        'final_bleu': bleus[-1],
                        'best_val_loss': min(test_losses),
                        'best_bleu': max(valid_bleus) if valid_bleus else 0.0
                    }
                    all_results.append(fold_stats)
                    print(f"Loaded existing results for Fold {fold_idx + 1}")
                except:
                    print(f"Warning: Could not load results for Fold {fold_idx + 1}")
    
    # Execute incomplete folds
    for fold_idx in folds_to_run:
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
        fold_info = f'Fold {fold_idx + 1}/{len(folds)}'
        train_losses, test_losses, bleus = run_training(
            model, train_iter, val_iter, loader, val_data, fold_dir, total_epochs, device, 
            periodic_func=periodic_func, fold_info=fold_info, bleu_every=bleu_every, save_every=save_every
        )
        
        # Save fold statistics
        # Filter out None values from bleus before finding max
        valid_bleus = [b for b in bleus if b is not None]
        fold_stats = {
            'fold': fold_idx + 1,
            'final_train_loss': train_losses[-1],
            'final_val_loss': test_losses[-1],
            'final_bleu': bleus[-1],
            'best_val_loss': min(test_losses),
            'best_bleu': max(valid_bleus) if valid_bleus else 0.0
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
    parser.add_argument('--functions', type=str, default=None,
                       help='Comma-separated list of periodic functions (default: all available or detect from existing)')
    parser.add_argument('--folds', type=str, default=None,
                       help='Comma-separated fold numbers to train (1-based, e.g., 1,2,3,4,5). Default: all folds')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (e.g., cuda:0, cuda:1). Overrides conf.py')
    parser.add_argument('--bleu-every', type=int, default=25,
                       help='Calculate BLEU every N epochs (default: 25). Set to 1 for every epoch.')
    parser.add_argument('--save-every', type=int, default=25,
                       help='Save losses and BLEU to files every N epochs (default: 25). Values are retained in memory.')
    args = parser.parse_args()
    
    # Configuration
    k_folds = 10
    epochs_per_fold = 1000
    
    # Determine periodic functions to use
    if args.functions:
        periodic_functions = [f.strip() for f in args.functions.split(',')]
    else:
        # Default list of functions
        periodic_functions = ['sinusoid', 'triangular', 'square', 'sawtooth']
    
    # Parse user-specified folds (convert from 1-based to 0-based)
    user_specified_folds = None
    if args.folds:
        try:
            # Convert from 1-based (user input) to 0-based (internal representation)
            user_specified_folds = [int(f.strip()) - 1 for f in args.folds.split(',')]
            # Validate fold numbers
            if any(f < 0 or f >= k_folds for f in user_specified_folds):
                print(f"Error: Fold numbers must be between 1 and {k_folds}")
                return
            print(f"User specified folds: {[f+1 for f in user_specified_folds]}")
        except ValueError:
            print("Error: --folds must be comma-separated integers (e.g., 1,2,3,4,5)")
            return
    
    # Set device (CLI argument overrides conf.py)
    global device
    if args.device:
        device = torch.device(args.device)
        print(f"Using device from CLI: {device}")
    else:
        print(f"Using device from conf.py: {device}")
    
    # Determine base directory and resume point
    if args.resume:
        base_dir = get_most_recent_results_dir()
        if base_dir is None:
            print("No previous results directory found. Starting new training.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = f'/mnt/home/users/tic_163_uma/macorisd/GitHub/alt-positional-encoding-transformer/results/results_{timestamp}'
            os.makedirs(base_dir, exist_ok=True)
        else:
            print(f"Found previous results: {base_dir}")
            
            # If --functions not specified, detect which functions exist in the results directory
            if not args.functions:
                existing_funcs = []
                for func in periodic_functions:
                    func_dir = os.path.join(base_dir, func)
                    if os.path.exists(func_dir):
                        existing_funcs.append(func)
                
                if existing_funcs:
                    periodic_functions = existing_funcs
                    print(f"Detected existing functions to resume: {', '.join(periodic_functions)}")
                else:
                    print(f"No existing functions found, will start with default: {', '.join(periodic_functions)}")
            
            # Check if all training already completed (considering user-specified folds)
            all_complete = True
            for periodic_func in periodic_functions:
                incomplete_folds = filter_folds_to_run(base_dir, periodic_func, k_folds, user_specified_folds)
                if incomplete_folds is not None:
                    all_complete = False
                    break
            
            if all_complete:
                if user_specified_folds:
                    print(f"All specified folds {[f+1 for f in user_specified_folds]} already completed!")
                else:
                    print("All training already completed!")
                return
            
            print(f"Will check resume point for each function individually...")
    else:
        # Create new base directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f'/mnt/home/users/tic_163_uma/macorisd/GitHub/alt-positional-encoding-transformer/results/results_{timestamp}'
        os.makedirs(base_dir, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f"# 10-FOLD CROSS VALIDATION - POSITIONAL ENCODING COMPARISON")
    print(f"# Timestamp: {os.path.basename(base_dir)}")
    print(f"# Output directory: {base_dir}")
    print(f"# K-folds: {k_folds}")
    print(f"# Epochs per fold: {epochs_per_fold}")
    print(f"# Periodic functions: {', '.join(periodic_functions)}")
    if user_specified_folds:
        print(f"# Specified folds: {[f+1 for f in user_specified_folds]}")
    if args.resume:
        print(f"# RESUME MODE: Will check each function individually")
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
    
    # If resuming, process each function individually
    if args.resume:
        for periodic_func in periodic_functions:
            # Check which folds need to be executed for this function
            folds_to_run = filter_folds_to_run(base_dir, periodic_func, k_folds, user_specified_folds)
            
            if folds_to_run is None:
                print(f"\n{'='*80}")
                print(f"SKIPPING {periodic_func}: All folds already completed")
                print(f"{'='*80}\n")
                
                # Load existing results for summary
                func_dir = os.path.join(base_dir, periodic_func)
                all_results = []
                for fold_idx in range(k_folds):
                    fold_dir = os.path.join(func_dir, f'{periodic_func}_{fold_idx + 1}')
                    if os.path.exists(fold_dir):
                        try:
                            with open(os.path.join(fold_dir, 'train_loss.txt'), 'r') as f:
                                train_losses = eval(f.read())
                            with open(os.path.join(fold_dir, 'test_loss.txt'), 'r') as f:
                                test_losses = eval(f.read())
                            with open(os.path.join(fold_dir, 'bleu.txt'), 'r') as f:
                                bleus = eval(f.read())
                            
                            valid_bleus = [b for b in bleus if b is not None]
                            fold_stats = {
                                'fold': fold_idx + 1,
                                'final_train_loss': train_losses[-1],
                                'final_val_loss': test_losses[-1],
                                'final_bleu': bleus[-1],
                                'best_val_loss': min(test_losses),
                                'best_bleu': max(valid_bleus) if valid_bleus else 0.0
                            }
                            all_results.append(fold_stats)
                        except:
                            pass
                all_function_results[periodic_func] = all_results
                continue
            
            print(f"\n{'='*80}")
            print(f"PROCESSING {periodic_func}: Executing folds {[f+1 for f in folds_to_run]}")
            print(f"{'='*80}\n")
            
            results = run_cross_validation(
                periodic_func, loader, folds, base_dir, epochs_per_fold, folds_to_run=folds_to_run, bleu_every=args.bleu_every, save_every=args.save_every
            )
            all_function_results[periodic_func] = results
    else:
        # New training, process all functions from the beginning
        for periodic_func in periodic_functions:
            results = run_cross_validation(
                periodic_func, loader, folds, base_dir, epochs_per_fold, folds_to_run=user_specified_folds, bleu_every=args.bleu_every, save_every=args.save_every
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
