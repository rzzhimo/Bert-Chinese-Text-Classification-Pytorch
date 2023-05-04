# coding: UTF-8
# 这个文件就是简单跑了一下测试集，并在跑测试集的过程中输出label和predict，可以直观地看出label和predict之间的区别
# 用法: set PYTHONIOENCODING=utf8 && python "相对路径/predict.py" --model bert
import time
import torch
import numpy as np
from train_eval import train, init_network,test
from tqdm import tqdm
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import logging

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def load_dataset(path, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):

            lin = line.strip().strip('\t')
            if not lin:
                continue
            if lin.find('\t')>0:
                content, label = lin.split('\t')
            else:
                content = lin
                label = -1
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))
    return contents

def predict(textList):
    return None

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    test_data = load_dataset(config.test_path, config.pad_size)
    print(test_data)
    test_iter = build_iterator(test_data, config)
    print(test_iter)
    time_dif = get_time_dif(start_time)


    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    print("Time usage:", time_dif)
    # train(config, model, train_iter, dev_iter, test_iter)
    #test(config,model,test_iter)
    print("start predict...")
    start_time = time.time()
    with torch.no_grad():
        for texts, lables in test_iter:
            print("label:")
            print(lables)

            outputs = model(texts)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            print("predict")
            print(predict)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)