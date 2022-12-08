"""
All Bert experiments

Usage:

    python3 language-models/bert.py grad
    python3 language-models/bert.py train
"""
import sys, argparse, pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model, utils, train
import language_models

PERMITTED_MODE = ['grad', 'filter-grad', 'train']

parser = argparse.ArgumentParser(
                prog = 'Bert Model',
                description = 'Experiments on the bert model',
                epilog = 'Bert')
parser.add_argument('mode', help=f"Execution type {','.join(PERMITTED_MODE)}")
parser.add_argument('-s', '--source', type=str,
                    default='medium', help='Source of bert model. {tiny,mini,small,medium}')
parser.add_argument('-t', '--topology', type=str,
                    default='all', help='Model parameter sharing scheme. {all, surgical, separate, shared}')
parser.add_argument('--device', type=str, default='cuda:1', help='Torch Device')
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
    ds_train, ds_val = language_models.get_dataset(name)

    generator_train = language_models.ClassificationDataGenerator(ds_train, tokenizer, args.device, batchSize=32)
    generator_val = language_models.ClassificationDataGenerator(ds_val, tokenizer, args.device, batchSize=32)
    return {
        'train_gen': generator_train,
        'val_gen': generator_val,
        'loss': loss_func, #lambda logits, labels: F.binary_cross_entropy_with_logits(logits, labels[..., None]),
        'predict': lambda logits: torch.sigmoid(logits) > 0.5,
        'metric': language_models.get_acc,
    }

tasks = {
    'mrpc': get_language_task('glue/mrpc'),
    'qnli': get_language_task('glue/qnli'),
}
task_keys = list(tasks.keys())

if mode == 'grad':
    modelShared = language_models.TaskAwareBert(bert, list(tasks.keys()),
                                                topology='shared',
                                                source=args.source).to(args.device)

    grads = train.get_gradients(
        model=modelShared,
        tasks=tasks,
        steps=200,
        lr=3e-4,
        DEVICE=args.device,
        param_keys=modelShared.backbone_trainables
    )
    # Condense the gradients
    def process_parameter_key(key):
        g0 = utils.stack_grad(grads, task_keys[0], key)
        g1 = utils.stack_grad(grads, task_keys[1], key)

        # heuristics computations
        cosine = torch.sum(F.normalize(g0, dim=-1) * F.normalize(g1, dim=-1), dim=-1).numpy()
        return cosine

    gradCosines = { k:process_parameter_key(k) for k in modelShared.backbone_trainables }

    with open(paths['plots'] / f'{args.source}-gradCosine.pickle', 'rb') as f:
        pickle.dump(gradCosines, f)
elif mode == 'filter_grad':
    print("Filter grad. Not implemented yet.")
elif mode == 'train':
    import torch.utils.tensorboard as TUTb

    topIterant = ['shared', 'separate', 'surgical'] \
        if args.topology == 'all' else [args.topology]

    for topology in topIterant:
        print(f"Using topology {topology}")
        model = language_models.TaskAwareBert(bert, list(tasks.keys()),
                                              topology=topology,
                                              source=args.source).to(args.device)
        writers = {k:TUTb.SummaryWriter(paths['logs'] / f"{args.source}-{topology}-{k}") for k in tasks.keys()}
        exp = train.train_and_evaluate(
            model=model,
            tasks=tasks,
            steps=500,
            lr=1e-4,
            eval_every=50,
            DEVICE=args.device,
            writers=writers,
        )
        with open(paths['plots'] / f"{args.source}-{topology}.pickle", "wb") as f:
            pickle.dump(exp, f)
