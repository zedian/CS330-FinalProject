"""
All Bert experiments

Usage:

    python3 language-models/bert.py grad
    python3 language-models/bert.py train
"""
import sys, argparse, pickle, itertools

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model, utils, train
import language_models

PERMITTED_MODE = ['grad', 'filter-grad', 'train']

SOURCES = ['tiny','mini','small','medium']

parser = argparse.ArgumentParser(
                prog = 'Bert Model',
                description = 'Experiments on the bert model',
                epilog = 'Bert')
parser.add_argument('mode', help=f"Execution type {','.join(PERMITTED_MODE)}")
parser.add_argument('-s', '--source', type=str,
                    default='medium',
                    help=f"Source of bert model: {','.join(SOURCES)}")
parser.add_argument('--steps', type=int,
                    default=0, help='Number of evaluation steps. Defaults to the longer of the two datasets')
parser.add_argument('-t', '--topology', type=str,
                    default='all', help='Model parameter sharing scheme. {all, surgical, separate, shared}')
parser.add_argument('--device', type=str, default='cuda:1', help='Torch Device')
parser.add_argument('--batchsize', type=int, default=192, help='Batch Size')
args = parser.parse_args()

mode = sys.argv[1]
assert args.mode in PERMITTED_MODE

paths = language_models.get_paths('bert')
bert, tokenizer = language_models.get_model_and_tokenizer(f'prajjwal1/bert-{args.source}')
stop_tokens = language_models.get_stop_tokens(tokenizer)

def loss_func(logits, labels):
    #print(logits, labels)
    return F.binary_cross_entropy_with_logits(logits, labels[..., None].float())
def get_language_task(name):
    ds_train, ds_val = language_models.get_dataset(name, tokenizer.sep_token)
    print(f"{name} has {len(ds_train)} train samples and {len(ds_val)} val samples")

    generator_train = language_models.ClassificationDataGenerator(ds_train, tokenizer, args.device, batchSize=args.batchsize)
    generator_val = language_models.ClassificationDataGenerator(ds_val, tokenizer, args.device, batchSize=args.batchsize)
    return {
        'train_gen': generator_train,
        'val_gen': generator_val,
        'loss': loss_func, #lambda logits, labels: F.binary_cross_entropy_with_logits(logits, labels[..., None]),
        'predict': lambda logits: torch.sigmoid(logits) > 0.5,
        'metric': language_models.get_acc,
    }

tasks = {
    'mrpc': get_language_task('glue/mrpc'),
    'sst2': get_language_task('glue/sst2'),
}
task_keys = list(tasks.keys())

if mode == 'grad':
    modelShared = language_models.TaskAwareBert(bert, list(tasks.keys()),
                                                topology='shared',
                                                source=args.source).to(args.device)

    cosines = train.get_gradients(
        model=modelShared,
        tasks=tasks,
        steps=args.steps,
        lr=3e-4,
        DEVICE=args.device,
        param_keys=modelShared.backbone_trainables
    )
    with open(paths['plots'] / f'{args.source}-gradCosine.pickle', 'wb') as f:
        pickle.dump(cosines, f)
elif mode == 'filter_grad':
    print("Filter grad. Not implemented yet.")
elif mode == 'train':
    import torch.utils.tensorboard as TUTb

    topIterant = ['shared', 'separate', 'surgical'] \
        if args.topology == 'all' else [args.topology]

    # Read on what to share
    with open(paths['plots'] / f'{args.source}-gradCosine.pickle', 'rb') as f:
        cosines, param_keys = pickle.load(f)

    param_keys = list(itertools.compress(param_keys, cosines.mean(axis=0) > 0))
    print(f"Sharing {len(param_keys)}/{cosines.shape[1]} layers...")

    for topology in topIterant:
        print(f"Using topology {topology}")
        model = language_models.TaskAwareBert(bert, list(tasks.keys()),
                                              topology=topology,
                                              source=args.source,
                                              sharing=param_keys)\
                               .to(args.device)
        writers = {k:TUTb.SummaryWriter(paths['logs'] / f"{args.source}-{topology}-{k}") for k in tasks.keys()}
        exp = train.train_and_evaluate(
            model=model,
            tasks=tasks,
            steps=args.steps,
            lr=6e-5,
            eval_every=50,
            DEVICE=args.device,
            writers=writers,
        )
        with open(paths['plots'] / f"{args.source}-{topology}.pickle", "wb") as f:
            pickle.dump(exp, f)
