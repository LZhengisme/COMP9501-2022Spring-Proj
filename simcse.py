#! -*- coding: utf-8 -*-
# SimCSE 中文测试

from utils import *
import sys
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
jieba.initialize()

#! -*- coding: utf-8 -*-
# 数据读取函数

from tqdm import tqdm
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding
from keras.models import Model


def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                D.append((l[0], l[1], float(l[2])))
    return D


def get_tokenizer(dict_path, pre_tokenize=None):
    """建立分词器
    """
    return Tokenizer(dict_path, do_lower_case=True, pre_tokenize=pre_tokenize)


def get_encoder(
    config_path,
    checkpoint_path,
    model='bert',
    pooling='first-last-avg',
    dropout_rate=0.1
):
    """建立编码器
    """
    assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']

    if pooling == 'pooler':
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            with_pool='linear',
            dropout_rate=dropout_rate
        )
    else:
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            dropout_rate=dropout_rate
        )

    outputs, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break

    if pooling == 'first-last-avg':
        outputs = [
            keras.layers.GlobalAveragePooling1D()(outputs[0]),
            keras.layers.GlobalAveragePooling1D()(outputs[-1])
        ]
        output = keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
    elif pooling == 'cls':
        output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
    elif pooling == 'pooler':
        output = bert.output

    # 最后的编码器
    encoder = Model(bert.inputs, output)
    return encoder


def convert_to_ids(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids, b_token_ids, labels


def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation



# 基本参数
model_type, pooling, task_name, dropout_rate = sys.argv[1:]
assert model_type in [
    'BERT', 'RoBERTa', 'NEZHA', 'WoBERT', 'RoFormer', 'BERT-large',
    'RoBERTa-large', 'NEZHA-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small'
]
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']
dropout_rate = float(dropout_rate)

if task_name == 'PAWSX':
    maxlen = 128
else:
    maxlen = 64

# 加载数据集
data_path = '/root/senteval_cn/'

datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['train', 'valid', 'test']
}

# bert配置
model_name = {
    'BERT': 'chinese_L-12_H-768_A-12',
    'RoBERTa': 'chinese_roberta_wwm_ext_L-12_H-768_A-12',
    'WoBERT': 'chinese_wobert_plus_L-12_H-768_A-12',
    'NEZHA': 'nezha_base_wwm',
    'RoFormer': 'chinese_roformer_L-12_H-768_A-12',
    'BERT-large': 'uer/mixed_corpus_bert_large_model',
    'RoBERTa-large': 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16',
    'NEZHA-large': 'nezha_large_wwm',
    'SimBERT': 'chinese_simbert_L-12_H-768_A-12',
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'
}[model_type]

config_path = '/root/kg/bert/%s/bert_config.json' % model_name
if model_type == 'NEZHA':
    checkpoint_path = '/root/kg/bert/%s/model.ckpt-691689' % model_name
elif model_type == 'NEZHA-large':
    checkpoint_path = '/root/kg/bert/%s/model.ckpt-346400' % model_name
else:
    checkpoint_path = '/root/kg/bert/%s/bert_model.ckpt' % model_name
dict_path = '/root/kg/bert/%s/vocab.txt' % model_name

# 建立分词器
if model_type in ['WoBERT', 'RoFormer']:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立模型
if model_type == 'RoFormer':
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='roformer',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
elif 'NEZHA' in model_type:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='nezha',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
else:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate
    )

# 语料id化
all_names, all_weights, all_token_ids, all_labels = [], [], [], []
train_token_ids = []
for name, data in datasets.items():
    a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)
    train_token_ids.extend(a_token_ids)
    train_token_ids.extend(b_token_ids)

if task_name != 'PAWSX':
    np.random.shuffle(train_token_ids)
    train_token_ids = train_token_ids[:10000]


class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)  # 每个batch里，每句话都重复了一次
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []


def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


# SimCSE训练
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
train_generator = data_generator(train_token_ids, 64)
encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=1
)

# 语料向量化
all_vecs = []
for a_token_ids, b_token_ids in all_token_ids:
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    all_vecs.append((a_vecs, b_vecs))

# 标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = l2_normalize(a_vecs)
    b_vecs = l2_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))