import argparse, copy
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

# Supplied function
import model
from typing import List, Optional, Tuple, Union



def str_strip(s):
    return s.replace(' .', '.')

    
    
def check_dataset_integrity(d):
    for s in d['x']:
        assert ' .' not in s, s
        assert '\n' not in s, s
        #assert '..' not in s, s
def get_dataset(dataset: str):
    
    
    if dataset == 'glue/cola':
        def transform_strip(row):
            s = row['sentence']
            s = str_strip(s)
            row['sentence'] = s
            return row
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
        def transform(row):
            row['sentence'] = row['sentence1'] + ' ' + row['sentence2']
            return row
            
        d_train = datasets.load_dataset('glue', 'mrpc', split='train')
        d_train = d_train.map(transform)
        d_train = d_train.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_train)
        d_val = datasets.load_dataset('glue', 'mrpc', split='validation')
        d_val = d_val.map(transform)
        d_val = d_val.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_val)
        return d_train, d_val
    
    if dataset == 'glue/qnli':
        def transform(row):
            row['sentence'] = row['question'] + ' ' + row['sentence']
            return row
        d_train = datasets.load_dataset('glue', 'qnli', split='train')
        d_train = d_train.map(transform)
        d_train = d_train.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_train)
        d_val = datasets.load_dataset('glue', 'qnli', split='validation')
        d_val = d_val.map(transform)
        d_val = d_val.rename_columns({'sentence': 'x', 'label': 'y'})
        #check_dataset_integrity(d_val)
        return d_train, d_val
    assert False, f"Unknown dataset: {dataset}"
    
class ClassificationDataGenerator:
    
    def __init__(self, dataset, tokenizer, device, batchSize=4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batchSize = batchSize
        
    def __len__(self):
        return len(self.dataset['y']) // self.batchSize
    
    def __getitem__(self, idx):
        batch = range(idx*self.batchSize, (idx+1)*self.batchSize)
        x = self.tokenizer([self.dataset['x'][i] for i in batch],
                           return_tensors='pt', padding=True, truncation=True, max_length=100).to(self.device)
        y = T.tensor([self.dataset['y'][i] for i in batch], device=self.device)
        return x,y
    
    def get_all(self, cutoff=None):
        x = self.tokenizer(self.dataset['x'] if cutoff is None else self.dataset[:cutoff]['x'],
                           return_tensors='pt', padding=True, truncation=True, max_length=100).to(self.device)
        y = T.tensor(self.dataset['y'] if cutoff is None else self.dataset[:cutoff]['y'],
                     device=self.device)
        return x,y
        
    
    
### Model Functions ###
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


def get_stop_tokens(tokenizer, stop_string: str = '.') -> int:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) == stop_string:
            tokens.append(idx)
    return tokens
    
class TaskAwareBert(TN.Module):
    
    def __init__(self, bert, tasks, topology):
        super().__init__()
        
        self.topology = topology
        
        assert topology in {'shared', 'separate', 'surgical'}
        
        if topology == 'shared':
            self.backbone = bert
            self.taskHeads = {
                task: TN.Linear(512, 1)
                for task in tasks
            }
            self.layers = TN.ModuleList(list(self.taskHeads.values()))
        elif topology == 'separate':
            self.taskHeads = {
                task: copy.deepcopy(bert)
                for task in tasks
            }
        elif topology == 'surgical':
            self.backbone = model.Shareable(
                mdl=bert,
                task_keys=tasks,
                shared_params=list(bert.state_dict().keys())
            )
            
    def forward(self, x, task):
        if self.topology == 'shared':
            logits = self.internal_forward(task, **x)
            return logits
        elif self.topology == 'separate':
            return self.taskHeads[task](**x).logits
        elif self.topology == 'surgical':
            return self.backbone(task, **x).logits
        
    def internal_forward(
        self,
        task: str,
        input_ids: Optional[T.Tensor] = None,
        attention_mask: Optional[T.Tensor] = None,
        token_type_ids: Optional[T.Tensor] = None,
        position_ids: Optional[T.Tensor] = None,
        head_mask: Optional[T.Tensor] = None,
        inputs_embeds: Optional[T.Tensor] = None,
        labels: Optional[T.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.backbone.config.use_return_dict

        outputs = self.backbone.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.backbone.dropout(pooled_output)
        return self.taskHeads[task](pooled_output)
    
### Training/Evaluation ###
def eval_model(model, tokenizer, ds_val, device):
    x = tokenizer(ds_val['x'], return_tensors='pt', padding=True, truncation=True, max_length=100).to(device)
    y = T.tensor(ds_val['y'], device=device)
    with T.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)
def finetune_selected(model, ds_train, ds_val, tokenizer, device='cuda', validationCutoff=500):
    
    model.to(device)
    val_acc = eval_model(model, tokenizer, ds_val, device)
    print(f"Before validation accuracy: {val_acc}")

    optimizer = TO.Adam(model_layer_select(model), lr=1e-4)
    
    generator_train = ClassificationDataGenerator(ds_train, tokenizer, device)
    
    xAll = tokenizer(ds_train['x'][:validationCutoff], return_tensors='pt', padding=True, truncation=True, max_length=100).to(device)
    yAll = T.tensor(ds_train['y'][:validationCutoff], device=device)
    
    iterant = tqdm.tqdm(generator_train)
    for step,(x_,y_) in enumerate(iterant):
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
            iterant.set_description(f'Fine-tuning acc: {total_acc:.04f}')
            
            # No thresholding accuracy
            #if total_acc > 0.75:
            #    break
    
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
    stop_tokens = get_stop_tokens(tokenizer)
    ds_train, ds_val = get_dataset('glue/mrpc')
    finetune_selected(model, ds_train, ds_val
                      , tokenizer, args.device)
    
    
    