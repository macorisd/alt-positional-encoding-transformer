"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# All text in the PDFs of the figures will be TrueType and the used subset of the fonts will be
# embedded. This will ensure the produced PDFs are acceptable for arXiv and other
# publication media
# See: https://github.com/matplotlib/matplotlib/issues/28521
plt.rcParams["pdf.fonttype"] = "truetype"


def read(name):
    """
    Reads a text file with comma-separated values and returns a list of floats or None values.
    """
    try:
        with open(name, 'r') as f:
            file = f.read()
            file = re.sub('\\[', '', file)
            file = re.sub('\\]', '', file)
            result = []
            for i in file.split(','):
                stripped = i.strip()
                if not stripped:
                    continue
                if stripped == 'None':
                    result.append(None)
                else:
                    result.append(float(stripped))
            return result
    except Exception as e:
        print(f"Error reading {name}: {e}")
        return []


def draw_loss(train_data, test_data, save_path, title='Training and Validation Loss'):
    """
    Generates loss graph for train and validation.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_data) + 1)
    plt.plot(epochs, train_data, 'r-', label='Train Loss', linewidth=2)
    plt.plot(epochs, test_data, 'b-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', axis='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Graph saved: {save_path}")


def draw_bleu(bleu_data, save_path, title='BLEU Score'):
    """
    Generates BLEU score graph.
    Handles None values for epochs where BLEU was not calculated.
    """
    plt.figure(figsize=(10, 6))
    
    # Filter out None values and keep track of corresponding epochs
    epochs_with_bleu = [i + 1 for i, b in enumerate(bleu_data) if b is not None]
    bleu_values = [b for b in bleu_data if b is not None]
    
    if not bleu_values:
        print(f"  Warning: No BLEU values to plot in {save_path}")
        plt.close()
        return
    
    # Add initial point (0, 0)
    epochs_with_bleu = [0] + epochs_with_bleu
    bleu_values = [0] + bleu_values
    
    plt.plot(epochs_with_bleu, bleu_values, 'go-', label='BLEU Score', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', axis='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Graph saved: {save_path}")


def process_fold(fold_path, fold_num, total_folds, wave_name):
    """
    Processes an individual fold: generates graphs if they don't exist and returns the data.
    Returns: (train_loss, test_loss, bleu, training_completed)
    """
    train_loss_file = fold_path / 'train_loss.txt'
    test_loss_file = fold_path / 'test_loss.txt'
    bleu_file = fold_path / 'bleu.txt'
    model_best_file = fold_path / 'model_best.pt'
    
    loss_graph = fold_path / 'loss_graph.pdf'
    bleu_graph = fold_path / 'bleu_graph.pdf'
    
    # Check if training has completed
    training_completed = model_best_file.exists()
    
    # Read data
    train_data = read(str(train_loss_file)) if train_loss_file.exists() else []
    test_data = read(str(test_loss_file)) if test_loss_file.exists() else []
    bleu_data = read(str(bleu_file)) if bleu_file.exists() else []
    
    # Generate graphs only if training has completed and graphs don't exist
    if training_completed:
        if not loss_graph.exists() and train_data and test_data:
            draw_loss(train_data, test_data, str(loss_graph), 
                     title=f'Training and Validation Loss - {wave_name} - Fold {fold_num}/{total_folds}')
        
        if not bleu_graph.exists() and bleu_data:
            draw_bleu(bleu_data, str(bleu_graph), 
                     title=f'BLEU Score - {wave_name} - Fold {fold_num}/{total_folds}')
    
    return train_data, test_data, bleu_data, training_completed


def process_wave(wave_path):
    """
    Processes a wave folder: processes all folds and generates average graphs.
    """
    wave_name = wave_path.name
    print(f"\nProcessing wave: {wave_name}")
    
    # Search for all fold folders
    fold_dirs = sorted([d for d in wave_path.iterdir() 
                       if d.is_dir() and d.name.startswith(f"{wave_name}_")])
    
    if not fold_dirs:
        print(f"  No fold folders found for {wave_name}")
        return
    
    print(f"  Folds found: {len(fold_dirs)}")
    
    # Process each fold and store data
    all_train_losses = []
    all_test_losses = []
    all_bleu_scores = []
    all_completed = []
    
    for idx, fold_dir in enumerate(fold_dirs, 1):
        print(f"  Processing fold: {fold_dir.name} ({idx}/{len(fold_dirs)})")
        train_loss, test_loss, bleu, completed = process_fold(fold_dir, idx, len(fold_dirs), wave_name)
        
        if train_loss:
            all_train_losses.append(train_loss)
        if test_loss:
            all_test_losses.append(test_loss)
        if bleu:
            all_bleu_scores.append(bleu)
        all_completed.append(completed)
    
    # Generate average graphs if all folds have completed
    loss_avg_graph = wave_path / 'loss_average_graph.pdf'
    bleu_avg_graph = wave_path / 'bleu_average_graph.pdf'
    
    if all(all_completed) and len(all_completed) > 0:
        print(f"  All folds completed for {wave_name}")
        
        # Generate average loss graph
        if not loss_avg_graph.exists() and all_train_losses and all_test_losses:
            # Calculate average (assuming all have the same length)
            min_len = min(len(lst) for lst in all_train_losses)
            train_avg = np.mean([lst[:min_len] for lst in all_train_losses], axis=0)
            test_avg = np.mean([lst[:min_len] for lst in all_test_losses], axis=0)
            
            draw_loss(train_avg, test_avg, str(loss_avg_graph),
                     title=f'Average Loss - {wave_name} ({len(fold_dirs)} folds)')
            
        # Generate average BLEU graph
        if not bleu_avg_graph.exists() and all_bleu_scores:
            min_len = min(len(lst) for lst in all_bleu_scores)
            # Average BLEU, handling None values
            bleu_lists = [lst[:min_len] for lst in all_bleu_scores]
            bleu_avg = []
            for i in range(min_len):
                epoch_values = [lst[i] for lst in bleu_lists if lst[i] is not None]
                if epoch_values:
                    bleu_avg.append(np.mean(epoch_values))
                else:
                    bleu_avg.append(None)
            
            draw_bleu(bleu_avg, str(bleu_avg_graph),
                     title=f'Average BLEU Score - {wave_name} ({len(fold_dirs)} folds)')
    else:
        completed_count = sum(all_completed)
        print(f"  Training in progress: {completed_count}/{len(fold_dirs)} folds completed")


def process_results_folder(results_base_path):
    """
    Processes all results folders that start with 'results_'.
    """
    base_path = Path(results_base_path)
    
    if not base_path.exists():
        print(f"Directory does not exist: {results_base_path}")
        return
    
    # Search for all folders starting with 'results_'
    results_dirs = sorted([d for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith('results_')])
    
    if not results_dirs:
        print(f"No 'results_*' folders found in {results_base_path}")
        return
    
    print(f"Found {len(results_dirs)} results folders")
    
    for results_dir in results_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {results_dir.name}")
        print(f"{'='*60}")
        
        # Search for wave folders (subfolders that don't start with 'results_')
        wave_dirs = sorted([d for d in results_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('results_')])
        
        if not wave_dirs:
            print(f"  No wave folders found in {results_dir.name}")
            continue
        
        print(f"  Waves found: {len(wave_dirs)}")
        
        for wave_dir in wave_dirs:
            process_wave(wave_dir)
    
    print(f"\n{'='*60}")
    print("Processing completed")
    print(f"{'='*60}")


def create_consolidated_graphs(results_base_path):
    """
    Creates two separate PDFs:
    - bleu_global_graph.pdf: BLEU scores for all 4 waves in different colors
    - loss_global_graph.pdf: Train (solid) and validation (dashed) losses for all 4 waves
    """
    base_path = Path(results_base_path)
    
    if not base_path.exists():
        print(f"Directory does not exist: {results_base_path}")
        return
    
    # Wave names and colors
    wave_names = ['sinusoid', 'triangular', 'square', 'sawtooth']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
    
    # Find the most recent results folder
    results_dirs = sorted([d for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith('results_')],
                         key=lambda x: x.name, reverse=True)
    
    if not results_dirs:
        print("No results folders found for consolidated graphs")
        return
    
    # Use the most recent results folder
    results_dir = results_dirs[0]
    print(f"\nCreating consolidated graphs from: {results_dir.name}")
    
    # Collect data for all waves
    waves_data = {}
    
    for wave_name in wave_names:
        wave_dir = results_dir / wave_name
        
        waves_data[wave_name] = {
            'train_losses': [],
            'test_losses': [],
            'bleu_scores': [],
            'bleu_interval': None  # Store the BLEU calculation interval
        }
        
        if not wave_dir.exists():
            print(f"  Wave directory not found: {wave_name}")
            continue
        
        # Find fold directories
        fold_dirs = sorted([d for d in wave_dir.iterdir() 
                           if d.is_dir() and d.name.startswith(f"{wave_name}_")])
        
        # Collect data from all folds
        all_train_losses = []
        all_test_losses = []
        all_bleu_scores = []
        
        for fold_dir in fold_dirs:
            train_loss_file = fold_dir / 'train_loss.txt'
            test_loss_file = fold_dir / 'test_loss.txt'
            bleu_file = fold_dir / 'bleu.txt'
            model_best = fold_dir / 'model_best.pt'
            
            # Only include if training is complete
            if model_best.exists():
                train_data = read(str(train_loss_file)) if train_loss_file.exists() else []
                test_data = read(str(test_loss_file)) if test_loss_file.exists() else []
                bleu_data = read(str(bleu_file)) if bleu_file.exists() else []
                
                if train_data:
                    all_train_losses.append(train_data)
                if test_data:
                    all_test_losses.append(test_data)
                if bleu_data:
                    all_bleu_scores.append(bleu_data)
        
        # Calculate averages
        if all_train_losses:
            min_len = min(len(lst) for lst in all_train_losses)
            waves_data[wave_name]['train_losses'] = np.mean([lst[:min_len] for lst in all_train_losses], axis=0)
        
        if all_test_losses:
            min_len = min(len(lst) for lst in all_test_losses)
            waves_data[wave_name]['test_losses'] = np.mean([lst[:min_len] for lst in all_test_losses], axis=0)
        
        if all_bleu_scores:
            min_len = min(len(lst) for lst in all_bleu_scores)
            # Average BLEU, handling None values
            bleu_lists = [lst[:min_len] for lst in all_bleu_scores]
            bleu_avg = []
            for i in range(min_len):
                epoch_values = [lst[i] for lst in bleu_lists if lst[i] is not None]
                if epoch_values:
                    bleu_avg.append(np.mean(epoch_values))
                else:
                    bleu_avg.append(None)
            waves_data[wave_name]['bleu_scores'] = bleu_avg
            
            # Calculate BLEU interval (every how many epochs BLEU is calculated)
            # Find the first non-None and then the next non-None to determine interval
            non_none_indices = [i for i, val in enumerate(bleu_avg) if val is not None]
            if len(non_none_indices) >= 2:
                waves_data[wave_name]['bleu_interval'] = non_none_indices[1] - non_none_indices[0]
            elif len(non_none_indices) == 1:
                waves_data[wave_name]['bleu_interval'] = 1
    
    # Generate BLEU graph
    plt.figure(figsize=(10, 6))
    
    for wave_name, color in zip(wave_names, colors):
        bleu_data = waves_data[wave_name]['bleu_scores']
        bleu_interval = waves_data[wave_name]['bleu_interval']
        
        if len(bleu_data) > 0:
            # Filter out None values and keep track of corresponding epochs
            epochs_with_bleu = [i + 1 for i, b in enumerate(bleu_data) if b is not None]
            bleu_values = [b for b in bleu_data if b is not None]
            
            if bleu_values:  # Only plot if there are valid values
                # Add initial point (0, 0)
                epochs_with_bleu = [0] + epochs_with_bleu
                bleu_values = [0] + bleu_values
                
                plt.plot(epochs_with_bleu, bleu_values, color=color, linestyle='-', 
                        marker='o', markersize=4, 
                        label=wave_name.capitalize(), linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('BLEU Score - All Waves', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', axis='both', alpha=0.3)
    plt.tight_layout()
    
    bleu_output_path = results_dir / 'bleu_global_graph.pdf'
    plt.savefig(bleu_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  BLEU graph saved: {bleu_output_path}")
    
    # Generate Loss graph
    plt.figure(figsize=(10, 6))
    
    for wave_name, color in zip(wave_names, colors):
        train_data = waves_data[wave_name]['train_losses']
        test_data = waves_data[wave_name]['test_losses']
        
        if len(train_data) > 0:
            epochs = range(1, len(train_data) + 1)
            plt.plot(epochs, train_data, color=color, linestyle='-', 
                    label=f'{wave_name.capitalize()} Train', linewidth=2)
        
        if len(test_data) > 0:
            epochs = range(1, len(test_data) + 1)
            plt.plot(epochs, test_data, color=color, linestyle='--', 
                    label=f'{wave_name.capitalize()} Val', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss - All Waves', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', axis='both', alpha=0.3)
    plt.tight_layout()
    
    loss_output_path = results_dir / 'loss_global_graph.pdf'
    plt.savefig(loss_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Loss graph saved: {loss_output_path}")
    
    print(f"\n{'='*60}")
    print(f"Consolidated graphs saved:")
    print(f"  - {bleu_output_path}")
    print(f"  - {loss_output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    results_path = script_dir / 'results'
    process_results_folder(results_path)
    create_consolidated_graphs(results_path)
