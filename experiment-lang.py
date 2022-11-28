import argparse
import numpy as N
import pandas
import torch as T
import torch.optim as TO
import torch.nn as TN
import torchvision as Tv
import torch.utils.data as TUD
import torch.nn.functional as TNF
import torchmetrics.functional as TmF
import pytorch_lightning as L
import pytorch_lightning.callbacks.progress as LCP
import datasets,transformers
import tqdm

#GPT2_MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']


# from a3
def get_model_and_tokenizer(modelName: str, Cls, **model_kwargs):

    m = Cls.from_pretrained(modelName, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(modelName)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({'pad_token': '[PAD]'})
            tok.pad_token = '[PAD]'
    return m, tok


def stop_tokens(tokenizer, stop_string: str = '.') -> int:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) == stop_string:
            tokens.append(idx)
    return tokens

def str_strip(s):
    return s.replace(' .', '.')
def transform_strip(row):
    s = row['sentence']
    s = str_strip(s)
    row['sentence'] = s
    return row
def transform_merge_sentence(row):
    row['sentence'] = row['sentence1'] + ' ' + row['sentence2']
    return row
    
    
def check_dataset_integrity(d):
    for s in d['x']:
        assert ' .' not in s, s
        assert '\n' not in s, s
        #assert '..' not in s, s
def get_dataset(dataset: str):
    
    
    if dataset == 'glue/cola':
        d_train = datasets.load_dataset('glue', 'cola', split='train')
        #d_train = d_train.map(transform_strip)
        d_train = d_train.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_train)
        d_val = datasets.load_dataset('glue', 'cola', split='validation')
        #d_val = d_val.map(transform_strip)
        d_val = d_val.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_val)
        return d_train, d_val
    
    if dataset == 'glue/mrpc':
        d_train = datasets.load_dataset('glue', 'mrpc', split='train')
        d_train = d_train.map(transform_merge_sentence)
        d_train = d_train.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_train)
        d_val = datasets.load_dataset('glue', 'mrpc', split='validation')
        d_val = d_val.map(transform_merge_sentence)
        d_val = d_val.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_val)
        return d_train, d_val
    
    assert False, f"Unknown dataset: {dataset}"
def model_layer_select(model):
    if model.name_or_path == 'prajjwal1/bert-medium':
        tunables = [x for x in model.modules() if type(x) == transformers.BertLayer]
        tunables = tunables[:5]
        parameters = [p for l in tunables for p in l.parameters()]
        return parameters
    
def get_loss(y, labels):
    return TNF.cross_entropy(y, labels)
def get_acc(y, labels):
    return TmF.accuracy(y.argmax(axis=-1), labels)
    
def eval_model(model, tokenizer, ds_val, device):
    x = tokenizer(ds_val['x'], return_tensors='pt', padding=True, truncation=True, max_length=100).to(device)
    y = T.tensor(ds_val['y'], device=device)
    with T.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)
def finetune_selected(model, ds_train, ds_val, tokenizer, device='cuda', validationCutoff=500):
    
    batchSize = 8
    model.to(device)
    val_acc = eval_model(model, tokenizer, ds_val, device)
    print(f"Before validation accuracy: {val_acc}")

    optimizer = TO.Adam(model_layer_select(model), lr=1e-4)
    xAll = tokenizer(ds_train['x'][:validationCutoff], return_tensors='pt', padding=True, truncation=True, max_length=100).to(device)
    yAll = T.tensor(ds_train['y'][:validationCutoff], device=device)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = N.random.randint(0, len(ds_train['y']), batchSize)
        x_ = tokenizer([ds_train['x'][i] for i in batch], return_tensors='pt', padding=True, truncation=True, max_length=100).to(device)
        #print(type(x_))
        y_ = T.tensor([ds_train['y'][i] for i in batch], device=device)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if args.debug:
            break

        if step % 10 == 0:
            with T.inference_mode():
                total_acc = get_acc(model(**xAll).logits, yAll)
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')
            if total_acc > 0.75:
                break
    
    val_acc = eval_model(model, tokenizer, ds_val, device)
    print(f"After validation accuracy: {val_acc}")

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog = 'Language Multi-tasking experiments',
        description = 'Trains GPT2')
    parser.add_argument('--model', default='prajjwal1/bert-medium')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    model, tokenizer = get_model_and_tokenizer(args.model, transformers.AutoModelForSequenceClassification)
    stop_tokens = stop_tokens(tokenizer)
    ds_train, ds_val = get_dataset('glue/mrpc')
    finetune_selected(model, ds_train, ds_val, tokenizer, args.device)
    
    
    