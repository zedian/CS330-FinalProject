## Setup

Export the environment variable `HUGGINGFACE_TOKEN` as an authentication token so some datasets can be loaded.


## Language Models

`$SOURCE` refers to source of bert model (`tiny`, `mini`, `small`, `medium`). All 4 must be executed
```bash
python3 experiments-bert.py grad --steps $STEPS --source $SOURCE
python3 experiments-bert.py train --topology all --source $SOURCE --steps $STEPS
```
For `RoBERTa` model, there is only one architecture
```bash
python3 experiments-roberta.py grad --steps $STEPS
python3 experiments-roberta.py train --topology all --steps $STEPS
```
