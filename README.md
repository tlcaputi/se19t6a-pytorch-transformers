# Sem Eval 2019 Task 6A
### An example using PyTorch, Transformers, and BERT to achieve State-of-the-Art results on a standard Twitter classification task.

### Motivation
This example is intended to replicate (reasonably closely) the submission by [Ping et al.](https://www.aclweb.org/anthology/S19-2011.pdf), who won the first prize in [Task 6A of the Sem Eval 2019 Competition](https://arxiv.org/abs/1903.08983) by achieving the highest MacroF score (0.829) on the Test Data with their BERT model. Ping et al. used [BERT's Tensorflow code](https://github.com/google-research/bert), but I use [PyTorch](https://pytorch.org/) and HuggingFace's [Transformers](https://github.com/huggingface/transformers) package. Further, Ping uses the default hyperparameter values, but I calibrate the model's hyperparameters using [hyperopt](http://hyperopt.github.io/hyperopt/) to maximize MacroF on the dev data.

### What This Does
Google Research's [Bidirectional Encoder Representations from Transformers (BERT)](https://github.com/google-research/bert) model revolutionized Natural Language Processing (NLP) and achieved state-of-the-art results in several important NLP tasks, including Sequence Classification. This example demonstrates how to fine-tune a pre-trained BERT model on the [OffensEval 2019](https://sites.google.com/site/offensevalsharedtask/offenseval2019) dataset using PyTorch and Transformers. It also provides a good template for fine-tuning BERT for other Twitter sequence classification tasks.

### How to Run
Clone this repo, then simply run `./run-se19.sh`. You'll need to have [pyenv](https://github.com/pyenv/pyenv) with Python version 3.6.10 installed to make the virtual environment work. It runs in around 10 minutes on an RTX 2070 Super.

### Results

The first three rows are the results presented in the Ping et al. paper. The last row is the result of this script. This script appears to modestly outperform Ping et al. in the optimized metric MacroF.

|                    | Test Data |        |
|--------------------|-----------|--------|
|                    | MacroF    | Acc    |
| Ping et al. Linear | 0.7558    | 0.8105 |
| Ping et al. LSTM   | 0.7501    | 0.7953 |
| Ping et al. BERT   | 0.8286    | **0.8628** |
| This Script (BERT)       | **0.9028**    | 0.8535 |