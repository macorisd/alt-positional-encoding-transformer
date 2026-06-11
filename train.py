"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import evaluate_bleu
from util.epoch_timer import epoch_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)



model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device,
                    periodic_func=periodic_func).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
print(f'Using positional encoding: {periodic_func}')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 factor=factor,
                                                 patience=patience)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
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
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """
    Evaluate model computing loss and BLEU score.
    Uses autoregressive generation for BLEU calculation.
    """
    model.eval()
    epoch_loss = 0
    
    # Calculate loss with teacher forcing (standard for loss computation)
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_flat)
            epoch_loss += loss.item()
    
    # Calculate BLEU with autoregressive generation
    trg_eos_idx = loader.target.vocab['<eos>']
    bleu_score = evaluate_bleu(
        model=model,
        iterator=iterator,
        vocab_target=loader.target,
        max_len=max_len,
        sos_idx=trg_sos_idx,
        eos_idx=trg_eos_idx,
        pad_idx=trg_pad_idx,
        device=device
    )
    
    return epoch_loss / len(iterator), bleu_score


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []  
    for step in range(total_epoch):
        start_time = time.time()  
        train_loss = train(model, train_iter, optimizer, criterion, clip)  
        valid_loss, bleu = evaluate(model, valid_iter, criterion)  
        end_time = time.time()  

        if step > warmup: 
            scheduler.step(valid_loss)  
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr}')

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)  

        if valid_loss < best_loss and step % 50 == 0:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)