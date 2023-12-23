import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from dataset import BilingualDataset, casual_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from tqdm import tqdm

import warnings

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# requirements.txt >>
# torch==1.8.1
# transformers==4.5.1
# datasets==1.6.2
# tokenizers==0.10.2
# pathlib==1.0.1
# <<


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    
    tokenizer_path = Path(config['tokenizer_path'].format('lang'))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            # vocab_size=config['vocab_size'],
            special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"],
            # explanation > https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.Trainers.WordLevelTrainer
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
    
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    # tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # train val split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    # from dataset.py > BilingualDataset
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f'Max Length of Source: {max_len_src}')
    print(f'Max Length of Target: {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False) # batch size 1 cause each sentence to be processed one at a time
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model']) # rest of parameters are already predefined in the function
    return model


    
def train_model(config):
    # defining device (if cuda or sometihng)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    #starting tensorboard to visualiE   
    writer = SummaryWriter(config['experimental_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Loading weights from {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id(['PAD']), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch: 02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # batch * seq_len
            decoder_input = batch['decoder_input'].to(device) # batch * seq_len
            encoder_mask = batch['encoder_mask'].to(device) # 0 * 1 * 1 * seq_len # only hide padding tokens
            decoder_mask = batch['decoder_mask'].to(device) # 0 * 1 * seq_len * seq_len  # also hide subsequent words
            
            # running through transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # batch * seq_len * d_model
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # batch * seq_len * d_model
            proj_output = model.proj(decoder_output) # batch * seq_len * vocab_tgt_len
            
            label = batch['label'].to(device) # batch * seq_len
            
            # transforms batch * seq_len * vocab_tgt_len  -->   batch * seq_len * vocab_tgt_len
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix({f"loss: ": f"{loss.item():6.3f}"})
            
            # logging to tensorboard
            writer.add_scaler('train_loss', loss.item(), global_step)
            writer.flush()
            
            #backprop
            loss.backward()
            
            # updating weights
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        # saving weights after each epoch
        model_filename = get_weights_file_path(config, f'{epoch: 02d}')
        torch.save(
            {
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }, model_filename
        )

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)