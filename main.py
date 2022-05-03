from datasets import load_dataset, load_metric
from datasets.utils import disable_progress_bar
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import argparse
import numpy as np
from sklearn.metrics import classification_report
import os
import re
import copy
import builtins as __builtin__
from torch import nn
from transformers import set_seed
disable_progress_bar()


# easy for debug
def print(*args, flush=True, **kwargs):
    __builtin__.print(*args, **kwargs, flush=flush)

def evaluate_metric(test_labels, test_preds, TASK):
    output_dict = classification_report(test_labels, test_preds, digits=3, output_dict=True)
    if TASK.startswith('stance'):
        res = (output_dict['1']['f1-score'] + output_dict['2']['f1-score']) / 2
    elif TASK == 'irony':
        res = output_dict['1']['f1-score']
    elif TASK == 'sentiment':
        res = output_dict['macro avg']['recall']
    else:
        res = output_dict['macro avg']['f1-score']
    return res

def pretty_print(d, col_size=10):
    if d is None or d == '{}':
        return
    hdr_fmt = ' | '.join(["{{:^{}}}".format(col_size) for _ in range(len(d))])
    val_fmt = ' | '.join(["{{:^{}.1f}}".format(col_size) for _ in range(len(d))])
    print(hdr_fmt.format(*d.keys()))
    print(val_fmt.format(*d.values()))

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_best_checkpoint(folder):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, min(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

def run_once(args, TASK_SPECS):
    metric_all_tasks = {}
    stance_all_preds = []
    stance_all_labels = []
    for TASK in args.tasks:
        if args.eval_only:
            MODEL = get_best_checkpoint(args.model + "-{}".format(TASK))
            TOKENIZER = "vinai/bertweet-base"
        else:
            MODEL = args.model # use this to finetune the language model
            TOKENIZER = MODEL
        dataset = load_dataset("tweet_eval", TASK.replace('-', '_'), cache_dir='.cache/')
        # dataset = load_dataset("yelp_review_full")


        output_dir  = args.output_dir  + '-' + args.model.replace('/', '-') + '-' + args.suffix + '-' + TASK
        logging_dir = args.logging_dir + '-' + args.model.replace('/', '-') + '-' + args.suffix + '-' + TASK
        training_args = TrainingArguments(
            output_dir=output_dir,                   # output directory
            num_train_epochs=TASK_SPECS[TASK]['num_epochs'],                  # total number of training epochs
            per_device_train_batch_size=args.batch_size,   # batch size per device during training
            per_device_eval_batch_size=int(args.batch_size * 8),    # batch size for evaluation
            learning_rate=TASK_SPECS[TASK]['lr'],
            warmup_ratio=0.1,                         # number of warmup steps for learning rate scheduler
            weight_decay=TASK_SPECS[TASK]['weight_decay'],                        # strength of weight decay
            logging_dir=logging_dir,                     # directory for storing logs
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            logging_strategy = 'epoch',
            load_best_model_at_end=True,              # load or not best model at the end
            save_total_limit=1,
            disable_tqdm=True,
            seed=args.seed,
        )
        if args.use_cl and not args.eval_only:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
            config = AutoConfig.from_pretrained(MODEL)
            config.num_labels = TASK_SPECS[TASK]['num_label']
            config.use_cl = True
            config.cl_lambda = args.cl_lambda
            config.cl_temp = args.cl_temp
            config.cl_scheme = args.cl_scheme
            model =  AutoModelForSequenceClassification.from_pretrained(MODEL, config=config)
            # optimizers = (create_optimizer(model, training_args), None)
            optimizers = (None, None)
        else:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=TASK_SPECS[TASK]['num_label'])
            optimizers = (None, None)
        if hasattr(model.config, 'max_position_embeddings'):
            max_length = model.config.max_position_embeddings - 2
        else:
            max_length = 512

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets["train"].shuffle(seed=args.seed)
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]

        metric = load_metric("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=model,                              # the instantiated 🤗 Transformers model to be trained
            args=training_args,                       # training arguments, defined above
            train_dataset=train_dataset,              # training dataset
            eval_dataset=val_dataset,                  # evaluation dataset
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        )

        if not args.eval_only:
            trainer.train()

        if args.eval_split == 'test':
            test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
        elif args.eval_split == 'validation':
            test_preds_raw, test_labels , _ = trainer.predict(val_dataset)
        test_preds = np.argmax(test_preds_raw, axis=-1)

        res = evaluate_metric(test_labels, test_preds, TASK)
        metric_all_tasks[TASK] = res * 100

        if TASK.startswith('stance'):
            stance_all_preds.append(test_preds)
            stance_all_labels.append(test_labels)
        # print(dataset['train'][1:5])

    stance_suite = ['stance-abortion', 'stance-atheism', 'stance-climate', 'stance-feminist',  'stance-hillary']
    stance_metrics = None
    if set(args.tasks).issuperset(set(stance_suite)):
        TASK = 'stance-all'
        stance_all_preds = np.concatenate(stance_all_preds)
        stance_all_labels = np.concatenate(stance_all_labels)
        # print("=================================> TASK : {} START <=================================".format(TASK))
        res = evaluate_metric(stance_all_labels, stance_all_preds, TASK)
        metric_all_tasks[TASK] = res * 100
        # print("=================================> TASK : {} END <=================================".format(TASK))

        stance_metrics = {}
        for key in stance_suite:
            stance_metrics[key] = metric_all_tasks.pop(key)
        stance_metrics['stance-all'] = metric_all_tasks['stance-all']
        metric_all_tasks['Avg'] = np.array(list(metric_all_tasks.values())).mean()
        

    return metric_all_tasks, stance_metrics

def main(args):
    set_seed(args.seed)
    print("========> The model will fine-tune (or evaluate) on the following tasks: <========", flush=True)
    print(str(args.tasks), flush=True)
    
    TASK_SPECS = {
        # best
        'emoji' : {
                    'num_label' : 20,
                    'lr' : 2.4e-5,
                    'weight_decay': 0.01,
                    'num_epochs' : 5,
                    },
        # best
        'emotion' : {
                    'num_label' : 4,
                    'lr' : 2e-5,
                    'weight_decay': 0.01,
                    'num_epochs' : 10,
                    },#
        # best
        'hate': {
                    'num_label' : 2,
                    'lr' : 2.4e-5,
                    'weight_decay': 0.01,
                    'num_epochs' : 5,
                    },
        # best
        'irony': {
                    'num_label' : 2,
                    'lr' : 5e-5,
                    'weight_decay': 0.05,
                    'num_epochs' : 5,
                    }, #
        'offensive' : {
                    'num_label' : 2,
                    'lr' : 2.4e-5,
                    'weight_decay': 0.01,
                    'num_epochs' : 5,
                    }, 
        # best
        'sentiment' : {
                    'num_label' : 3,
                    'lr' : 1e-5,
                    'weight_decay': 0.1,
                    'num_epochs' : 5,
                    },
        # best
        'stance-abortion' : {
                    'num_label' : 3,
                    'lr' : 3e-5,
                    'weight_decay': 0.05,
                    'num_epochs' : 20,
                    }, #
        'stance-atheism' : {
                    'num_label' : 3,
                    'lr' : 3e-5,
                    'weight_decay': 0.05,
                    'num_epochs' : 20,
                    }, #
        'stance-climate' : {
                    'num_label' : 3,
                    'lr' : 3e-5,
                    'weight_decay': 0.05,
                    'num_epochs' : 20,
                    }, #
        'stance-feminist' : {
                    'num_label' : 3,
                    'lr' : 3e-5,
                    'weight_decay': 0.05,
                    'num_epochs' : 20,
                    }, #
        'stance-hillary' : {
                    'num_label' : 3,
                    'lr' : 3e-5,
                    'weight_decay': 0.05,
                    'num_epochs' : 20,
                    },#
    }

    metric_all_tasks, stance_metrics = run_once(args, TASK_SPECS)
    pretty_print(metric_all_tasks)
    pretty_print(stance_metrics, col_size=17)

def get_args_parser():
    parser = argparse.ArgumentParser('Twitter Classification with Supervised Contrasts training and evaluation script', add_help=False)
    parser.add_argument(
        '--tasks', 
        nargs='+', 
        default=['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment',
            'stance-abortion', 'stance-atheism', 'stance-climate', 'stance-feminist',  'stance-hillary'])
    parser.add_argument('--model', default='vinai/bertweet-base', type=str, metavar='MODEL',
                        help='Name of model to train or evaluate')
    parser.add_argument('--eval-only', action='store_true', default=False)
    parser.add_argument('--eval-split', type=str, default='test', help='which split to perform evaluation')
    parser.add_argument('--suffix', type=str, default='', help='suffix for logging and output.')
    parser.add_argument('--use-cl', action='store_true', default=False)
    parser.add_argument('--cl-scheme', type=str, default='scl', help='suffix for logging and output.')
    parser.add_argument('--cl-lambda', type=float, default=0.5)
    parser.add_argument('--cl-temp', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--logging-dir', type=str, default='./logs')
    parser.add_argument('--seed', default=9501, type=int)
    
    return parser

if __name__ == "__main__":
    # print(get_last_checkpoint('results-bert-base-uncased-sentiment'))
    # exit(0)
    parser = argparse.ArgumentParser('COMP 9501 course project script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)