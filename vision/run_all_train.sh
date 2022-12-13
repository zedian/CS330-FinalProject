### CIFAR10 CIFAR10C
## ResNet18
# python train_cifar.py --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --model resnet18 --cifarcsmall True
# python train_cifar.py --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --model resnet18 --pretrained True --cifarcsmall True

# python train_cifar.py --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --model resnet18
# python train_cifar.py --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --model resnet18 --pretrained True
# Mixer
# python train_cifar.py --n_step 2000 --data_t0 cifar10 --data_t1 cifar10c --model mixer --cifarcsmall True

python train_cifar.py --n_step 2000 --data_t0 cifar10 --data_t1 cifar10c --model mixer
# Imagenet ResNet18
# python train_cifar.py --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --model imagenetresnet18 --cifarcsmall True

# python train_cifar.py --n_step 500 --data_t0 cifar10 --data_t1 cifar10c --model imagenetresnet18

### CIFAR10C CIFAR10CF
## ResNet18
# python train_cifar.py --n_step 500 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18 --cifarcsmall True
# python train_cifar.py --n_step 500 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18 --pretrained True --cifarcsmall True

# python train_cifar.py --n_step 500 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18
# python train_cifar.py --n_step 500 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18 --pretrained True
# Mixer
# python train_cifar.py --n_step 2000 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --cifarcsmall True

# python train_cifar.py --n_step 2000 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --mixer_depth 4
python train_cifar.py --n_step 2000 --data_t0 cifar10c --data_t1 cifar10cf --model mixer
python train_cifar.py --n_step 2000 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --mixer_depth 16
# Imagenet ResNet18
# python train_cifar.py --n_step 500 --data_t0 cifar10c --data_t1 cifar10cf --model imagenetresnet18 --cifarcsmall True

# python train_cifar.py --n_step 500 --data_t0 cifar10c --data_t1 cifar10cf --model imagenetresnet18
