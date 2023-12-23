from pathlib import Path

def get_config():
    return{
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload":None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experimental_name": "runs/tmodel",
        "seq_len": 350,
    }

def get_weights_file_path(config, epoch:str):
    
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    
    return str(Path('.') / (model_folder) / model_filename)
