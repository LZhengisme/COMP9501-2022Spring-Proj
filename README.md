# Course Project for COMP 9501

Group Member:
- Qintong Li(uid: 3030084294) and Lin Zheng(uid: 3030083422).

## Installation
Our implementation is built upon the 🤗 [Huggingface transformers](https://github.com/huggingface/transformers) codebase. To install all the dependencies, you need to first create a virtual Python environment and follow the [official instructions](https://pytorch.org/) to install the latest version of **PyTorch**. And then run
```bash
pip3 install numpy scikit-learn datasets
cd transformers
python3 setup.py build develop --user
cd ..
```

## Description
```bash
python3 main.py --tasks <a list of tasks> <--eval-only> --model <model-name> --batch-size <batch-size> --output-dir <output-dir> --logging-dir <logging-dir>  --eval-split <split> --suffix <path-suffix> <--use-cl>
--cl-scheme <cl-scheme> --cl-lambda <cl-lambda> --cl-temp <cl-temp>
```
- The usage of each argument used above can be recognized easily from its name, where
  - `--tasks`: accepts a list of tasks that you want to fine-tune and evaluate,
  - `--eval-only`: if set, the script will only do the evaluation step and skip any training;
  - `model-name` specifies the name of the model. default to our baseline BERTweet `vinai/bertweet-base`;
  - `batch-size` are used in fine-tuning pretrained models;
  - `output-dir`: specifies the directory containing all model outputs;
  - `logging-dir`: specifies the directory containing all log files;
  - `split` could be either `validation` or `test` to evaluate the model performance on the corresponding dataset split.
  - `--suffix`: specifies the model path suffix in case you want to annotate something;
  - `--use-cl`: use contrastive loss in training;
  - `--cl-scheme`: use either `scl` or `dscl`;
  - `--cl-lambda` and `--cl-temp` are the loss coefficient and temperature hyper-parameters respectively.
More information can be found in the script `main.py`.


## Fine-tuning
Fine-tuning BERTweet on all tasks with dscl loss can be done via the following,
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
     --tasks emoji --model vinai/bertweet-base --use-cl --cl-scheme dscl --cl-temp 0.3 --cl-lambda 0.5 \ 
     --batch-size 32 --eval-split test
CUDA_VISIBLE_DEVICES=1 python3 main.py  \
     --tasks emotion --model vinai/bertweet-base --use-cl --cl-scheme dscl --cl-lambda 0.3 --cl-temp 0.5 --suffix dscl --batch-size 32 \
     --eval-split test
CUDA_VISIBLE_DEVICES=2 python3 main.py  \
     --tasks hate --model vinai/bertweet-base --use-cl --cl-scheme dscl --cl-lambda 0.3 --cl-temp 0.5 --suffix dscl --batch-size 32 \
     --eval-split test
CUDA_VISIBLE_DEVICES=3 python3 main.py  \
     --tasks irony --model vinai/bertweet-base --use-cl --cl-scheme dscl --cl-lambda 0.3 --cl-temp 0.5 --suffix dscl --batch-size 32 \
     --eval-split test
CUDA_VISIBLE_DEVICES=4 python3 main.py  \
     --tasks offensive --model vinai/bertweet-base --use-cl --cl-scheme dscl --cl-lambda 0.3 --cl-temp 0.5 --suffix dscl --batch-size 32 \
     --eval-split test
CUDA_VISIBLE_DEVICES=5 python3 main.py \
     --tasks sentiment --model vinai/bertweet-base --use-cl --cl-scheme dscl --cl-temp 0.3 --cl-lambda 0.5 --suffix dscl --batch-size 32 \
     --eval-split test
CUDA_VISIBLE_DEVICES=6 python3 main.py  \
     --tasks stance-abortion stance-atheism stance-climate stance-feminist stance-hillary \
     --model vinai/bertweet-base --use-cl --cl-scheme dscl --cl-lambda 0.3 --cl-temp 0.5 --suffix dscl --batch-size 32 \
     --eval-split test
```

## Evaluation
Evaluation could be done via the following command:

To reproduce our results, please run the following example commands:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model <your-model-path> --eval-only --eval-split validation \
     --output-dir ./val-results --logging-dir ./val-logs 
CUDA_VISIBLE_DEVICES=1 python3 main.py --model <your-model-path> --eval-only --eval-split test \
     --output-dir ./test-results --logging-dir ./test-logs
```
The datasets and pretrained models will be automatically downloaded and cached locally.