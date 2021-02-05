# Hateful Memes Competition

![alt text](https://drivendata-public-assets.s3.amazonaws.com/memes-overview.png)

Take an image, add some text: you've got a meme. Internet memes are often harmless and sometimes hilarious. However, by using certain types of images, text, or combinations of each of these data modalities, the seemingly non-hateful meme becomes a multimodal type of hate speech, a hateful meme.

This repository is an implementation of the Hateful Memes detection competition where the goal is to build a binary classifier for detecting hateful memes over the internet using an annotated data set.

Competition details:
https://www.drivendata.org/competitions/64/hateful-memes/

## Setup Instructions

Create a new Python 3.7 virtual environment:

```
python3.7 -m venv .venv
```

Activate the virtual environment:

```
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```


Add the project directory to python PATH, in `~/.bashrc` or `~/.zshrc`:
```
export PYTHONPATH=/your/path/to/hateful-memes:$PYTHONPATH
```

Run black:

```
make black_check
```

## Training Instructions
The project is based on AllenNLP library as an infrastructure for training.

One should create a new config file or use one from the already existing config files inside `training_configs` and run the following command, for example:

```
python main_allennlp.py train training_configs/hateful_memes_bert.jsonnet --include-package src -s experiments/your_experiment
