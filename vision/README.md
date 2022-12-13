# Vision experiments on CIFAR10

```
# Accumulate gradients
python grad_cifar.py --model resnet18 --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --pretrained True --batch_size 256 --lr 0.0001
# Train
python train_cifar.py --model resnet18 --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --pretrained True --batch_size 256 --lr 0.0001
```

Accumulating gradients creates the directory

Before running train, make sure to include a param_keys.txt file in the directory associated to your experiment.
