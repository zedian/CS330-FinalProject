### CIFAR10 CIFAR10C
## ResNet18
# python grad_cifar.py --n_step 250 --data_t0 cifar10 --data_t1 cifar10c --model resnet18 --cifarcsmall True
# python grad_cifar.py --n_step 250 --data_t0 cifar10 --data_t1 cifar10c --model resnet18 --pretrained True --cifarcsmall True

# python grad_cifar.py --n_step 250 --data_t0 cifar10 --data_t1 cifar10c --model resnet18
# python grad_cifar.py --n_step 250 --data_t0 cifar10 --data_t1 cifar10c --model resnet18 --pretrained True
# Mixer
python grad_cifar.py --n_step 1500 --data_t0 cifar10 --data_t1 cifar10c --model mixer --cifarcsmall True
python grad_cifar.py --n_step 1500 --data_t0 cifar10 --data_t1 cifar10c --model mixer --cifarcsmall True --mixer_depth 8
python grad_cifar.py --n_step 1500 --data_t0 cifar10 --data_t1 cifar10c --model mixer --cifarcsmall True --mixer_depth 16

python grad_cifar.py --n_step 1500 --data_t0 cifar10 --data_t1 cifar10c --model mixer
python grad_cifar.py --n_step 1500 --data_t0 cifar10 --data_t1 cifar10c --model mixer --mixer_depth 8
python grad_cifar.py --n_step 1500 --data_t0 cifar10 --data_t1 cifar10c --model mixer --mixer_depth 16
# Imagenet ResNet18
# python grad_cifar.py --n_step 250 --data_t0 cifar10 --data_t1 cifar10c --model imagenetresnet18 --cifarcsmall True

# python grad_cifar.py --n_step 250 --data_t0 cifar10 --data_t1 cifar10c --model imagenetresnet18

### CIFAR10C CIFAR10CF
## ResNet18
# python grad_cifar.py --n_step 250 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18 --cifarcsmall True
# python grad_cifar.py --n_step 250 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18 --pretrained True --cifarcsmall True

# python grad_cifar.py --n_step 250 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18
# python grad_cifar.py --n_step 250 --data_t0 cifar10c --data_t1 cifar10cf --model resnet18 --pretrained True
# Mixer
python grad_cifar.py --n_step 1500 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --cifarcsmall True
python grad_cifar.py --n_step 1500 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --cifarcsmall True --mixer_depth 8
python grad_cifar.py --n_step 1500 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --cifarcsmall True --mixer_depth 16

python grad_cifar.py --n_step 1500 --data_t0 cifar10c --data_t1 cifar10cf --model mixer
python grad_cifar.py --n_step 1500 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --mixer_depth 8
python grad_cifar.py --n_step 1500 --data_t0 cifar10c --data_t1 cifar10cf --model mixer --mixer_depth 16
# Imagenet ResNet18
# python grad_cifar.py --n_step 250 --data_t0 cifar10c --data_t1 cifar10cf --model imagenetresnet18 --cifarcsmall True

# python grad_cifar.py --n_step 250 --data_t0 cifar10c --data_t1 cifar10cf --model imagenetresnet18
