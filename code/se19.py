from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import json
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch
from typing import *
from torch.utils.data.dataset import Dataset
from filelock import FileLock
import re
import mysql.connector as mysql
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    DataProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    InputExample,
    InputFeatures,
    PreTrainedTokenizer,
)
import wordsegment
import emoji
import time
from sklearn.model_selection import train_test_split
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ROOTPATH", default="", help="Path of root directory")
args = parser.parse_args()
ROOTPATH = args.ROOTPATH

def twitter_bert(
    ROOTPATH=ROOTPATH,
    model_name_or_path="bert-base-uncased",
    task_name="TWIT",
    do_train=True,
    do_eval=True,
    data_dir=f'{ROOTPATH}/input',
    max_seq_length=128,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3.0,
    cache_dir=None,
    output_dir=f'{ROOTPATH}/output',
    overwrite_cache=True,
    overwrite_output_dir=True,
    local_rank=-1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    n_gpu=torch.cuda.device_count(),
    fp16=False,
    num_labels=2,
    evaluate_during_training=False,
    weight_decay=0,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    train_dataset=None,
    dev_dataset=None,
    test_dataset=None,
    full_dataset=None,
    labels=None,
    temp_json=f'{ROOTPATH}/temp/run{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    use_test=False,
    save_steps=1e200,
    random_state=1234
):

    set_seed(random_state)
    if full_dataset is not None:
        train_dataset, dev_dataset = train_test_split(full_dataset, test_size=0.2, random_state=random_state)

    # Setup logging
    logger = logging.getLogger(__name__)

    logger.info(f"LENGTH OF TRAIN DATASET: {len(train_dataset.index)}")
    # exit(0)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        device,
        n_gpu,
        bool(local_rank != -1),
        fp16,
    )

    logger.info("Training/evaluation parameters local_rank: %s, device: %s, n_gpu: %s, fp16: %s", local_rank, device, n_gpu, fp16)
    logger.info(f"MAX SEQ LEN: {max_seq_length}")

    wordsegment.load()

    ## DEFINE FUNCTIONS
    @dataclass
    class ModelArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
        """

        model_name_or_path: str = field(
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
        config_name: Optional[str] = field(
            default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
        )
        tokenizer_name: Optional[str] = field(
            default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
        )
        cache_dir: Optional[str] = field(
            default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        overwrite_output_dir=overwrite_output_dir,
        evaluate_during_training=evaluate_during_training,
        weight_decay=weight_decay,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,
        save_steps=save_steps
    )

    data_args = DataTrainingArguments(
        task_name=task_name,
        data_dir=data_dir,
        max_seq_length=max_seq_length,
        overwrite_cache=overwrite_cache
    )

    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
    )

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def compute_metrics(preds, labels):
        assert len(preds) == len(labels)
        return acc_and_f1(preds, labels)

    class TwitterProcessor(DataProcessor):

        def __init__(self):

            super(TwitterProcessor, self).__init__()

            '''
            You need to define three variables here:
            - self.train_dataset -> train dataset
            - self.dev_dataset -> dev dataset
            - self.test_dataset -> test dataset
            - self.labels -> a list of the labels

            Each {train,dev,test}_dataset must have (at least) two columns:
            - "tweet" -> includes the text of the tweet
            - "label" -> includes the label of the tweet


            '''

            self.train_dataset = train_dataset
            self.dev_dataset = dev_dataset
            self.test_dataset = test_dataset
            self.labels = labels

        def get_train_examples(self):
            return self._create_examples(self.train_dataset, "train")

        def get_dev_examples(self):
            return self._create_examples(self.dev_dataset, "train")

        def get_test_examples(self):
            return self._create_examples(self.test_dataset, "train")

        def get_labels(self):
            """See base class."""
            return self.labels

        def _preprocess_text(self, text):
            # 1
            text = emoji.demojize(text)

            # 2
            words = text.split()
            for word in words:
                if word[0] != '#':
                    continue
                hashtag = word[1:]
                replacement_words = wordsegment.segment(hashtag)
                text = text.replace(word, " ".join(replacement_words))

            # 3
            text = text.replace("URL", "http")

            # 4
            text = re.sub(r'(@[A-Za-z]+)( \1\b){3,}', r'\1 \1 \1', text)
            return text

        def _create_examples(self, data, set_type):

            examples = []

            raw_texts = data.tweet.values.tolist()
            raw_labels = data.label.values.tolist()

            for i in range(0, len(raw_texts)):
                guid = "%s-%s" % (set_type, i)
                raw_text = raw_texts[i]
                raw_label = raw_labels[i]
                label = raw_label

                text = self._preprocess_text(raw_text)
                examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

            return examples

    def convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
    ):
        if max_length is None:
            max_length = tokenizer.max_len

        processor = TwitterProcessor()
        label_list = processor.get_labels()

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample) -> Union[int, float, None]:
            return label_map[example.label]

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features[i])

        return features

    class TwitterDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        def __init__(
            self,
            tokenizer,
            mode="train",
            cache_dir=cache_dir,
            args=data_args,
        ):
            self.args = args
            self.processor = TwitterProcessor()
            self.output_mode = 'Classification'

            label_list = self.processor.get_labels()
            self.label_list = label_list

            if mode == "dev":
                examples = self.processor.get_dev_examples()
            elif mode == "test":
                examples = self.processor.get_test_examples()
            else:
                examples = self.processor.get_train_examples()

            self.features = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=max_seq_length,
                label_list=label_list,
                output_mode=self.output_mode,
            )

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

        def get_labels(self):
            return self.label_list

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return compute_metrics(preds, p.label_ids)

        return compute_metrics_fn

    # Create model
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
    )

    # Get datasets
    train_dataset = (
        TwitterDataset(tokenizer=tokenizer, mode="train", cache_dir=cache_dir)
    )
    eval_dataset = (
        TwitterDataset(tokenizer=tokenizer, mode="dev", cache_dir=cache_dir)
    )

    if use_test:
        test_dataset = (
            TwitterDataset(tokenizer=tokenizer, mode="test", cache_dir=cache_dir)
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(task_name),
    )

    # Train the model
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model(f"{training_args.output_dir}/{task_name}")
        tokenizer.save_pretrained(f"{training_args.output_dir}/{task_name}")

    # Evaluation
    eval_results = []
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if use_test:
            step_names = ["dev", "test"]
            eval_datasets = [eval_dataset, test_dataset]
        else:
            step_names = ["dev"]
            eval_datasets = [eval_dataset]

        ct = 0
        for eval_dataset in eval_datasets:

            step_name = step_names[ct]

            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            logger.info("***** Eval results {} - {}*****".format(eval_dataset.args.task_name, step_name.upper()))
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)

            # output_eval_file = os.path.join(
            #     training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}_{step_name}.txt"
            # )

            # if ct == 0:
            #     with open(output_eval_file, "w") as writer:
            #         logger.info("***** Eval results {} - {}*****".format(eval_dataset.args.task_name, step_name.upper()))
            #         for key, value in eval_result.items():
            #             logger.info("  %s = %s", key, value)
            # else:
            #     with open(output_eval_file, "a") as writer:
            #         logger.info("***** Eval results {} - {}*****".format(eval_dataset.args.task_name, step_name.upper()))
            #         for key, value in eval_result.items():
            #             logger.info("  %s = %s", key, value)

            eval_results.append(eval_result)

            write_type = 'a' if os.path.exists(temp_json) else 'w'
            with open(temp_json, write_type) as f:
                f.write(json.dumps(eval_result))
                f.write("\n")

            ct += 1

    return eval_results[-1]


# ANALYSIS
if __name__ == "__main__":

    np.random.seed(1234)
    task_name = "se19hyperopt"
    data_dir = f"{ROOTPATH}/data"

    training = pd.read_csv(
        os.path.join(data_dir, "olid-training-v1.0.tsv"),
        sep="\t"
    )

    training = training.rename(columns={'subtask_a': 'label'})

    train_dataset, dev_dataset = train_test_split(training, test_size=0.1, random_state=1234)

    test_labels = pd.read_csv(
        os.path.join(data_dir, "labels-levela.csv"),
        names=['id', 'label']
    )

    test_text = pd.read_csv(
        os.path.join(data_dir, "testset-levela.tsv"),
        sep="\t"
    )

    test_dataset = pd.DataFrame({
        'tweet': test_text.tweet.values.tolist(),
        'label': test_labels.label.values.tolist()
    })

    labels = ["OFF", "NOT"]

    space = {
        'max_seq_length': scope.int(hp.quniform('max_seq_length', 100, 250, 10)),
        'learning_rate': hp.uniform('learning_rate', 1e-6, 1e-4),
        'per_device_train_batch_size': scope.int(hp.quniform('per_device_train_batch_size', 4, 10, 2)),
        'per_device_eval_batch_size': scope.int(hp.quniform('per_device_eval_batch_size', 4, 10, 2)),
        'num_train_epochs': scope.int(hp.quniform('num_train_epochs', 1, 6, 1)),
    }

    temp_json = f'{ROOTPATH}/temp/run{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{task_name}-test.json'
    
    def obj_fnc(params):

        write_type = 'a' if os.path.exists(temp_json) else 'w'
        with open(temp_json, write_type) as f:
            f.write(json.dumps(params))
            f.write("\n")

        res = twitter_bert(
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            labels=labels,
            task_name=task_name,
            temp_json=temp_json,
            use_test=False,
            **params
        )
        return -res['eval_f1']


    # UNCOMMENT THIS TO OPTIMIZE HYPERPARAMETERS, WILL TAKE LONGER 

    # hypopt_trials = Trials()
    # best_params = fmin(obj_fnc, space, algo=tpe.suggest, max_evals=10, trials=hypopt_trials)
    # out = {"Accuracy": hypopt_trials.best_trial['result']['loss'], "Best params" : best_params}

    # with open(f'{ROOTPATH}/output/outputstats-{task_name}.json', 'w') as f:
    #     f.write(json.dumps(out))


    best_params = {'learning_rate': 5.9742623354212456e-06, 'max_seq_length': 140, 'num_train_epochs': 1, 'per_device_eval_batch_size': 4, 'per_device_train_batch_size': 8}
    best_params.update(
        max_seq_length=int(best_params.get('max_seq_length')),
        num_train_epochs=int(best_params.get('num_train_epochs')),
        per_device_eval_batch_size=int(best_params.get('per_device_eval_batch_size')),
        per_device_train_batch_size=int(best_params.get('per_device_train_batch_size')),
    )
    res = twitter_bert(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        labels=labels,
        task_name=task_name,
        temp_json=temp_json,
        use_test=True,
        **best_params
    )
    print(best_params)
    print(res)
