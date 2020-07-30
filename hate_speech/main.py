import sys
import json
import math
import typing
from typing import Dict, List
import os
from argparse import ArgumentParser
import random
from torch import nn, optim
import torch
from torchtext.data import Iterator
from tqdm import tqdm
import numpy as np
import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from torchtext.data import Example
from model import BaseLine
from data import HateSpeech
from sklearn.metrics import recall_score, precision_score, f1_score


def bind_model(model):
    def save(dirname, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        model.load_state_dict(checkpoint['model'])

    def infer(raw_data, **kwargs):
        model.eval()
        examples = HateSpeech(raw_data).examples
        tensors = [torch.tensor(ex.syllable_contents, device='cuda').reshape([-1, 1]) for ex in examples]
        results = [model(ex).tolist() for ex in tensors]
        return results

    nsml.bind(save=save, load=load, infer=infer)


# models = [model0, model1, ...]
def bind_models(models):  # for ensemble
    def save(dirname, *args):
        checkpoint = {
            f'model{i}': model.state_dict() for i, model in enumerate(models)
        }
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        for i, model in enumerate(models):
            model.load_state_dict(checkpoint[f'model{i}'])

    def infer(raw_data, **kwargs):
        for model in models:
            model.eval()
        examples = HateSpeech(raw_data).examples
        tensors = [torch.tensor(ex.syllable_contents, device='cuda').reshape([-1, 1]) for ex in examples]
        results = []
        for ex in tensors:
            total = 0
            for model in models:  # vote
                total += torch.where(model(ex) > 0.5, torch.tensor(1.0, device='cuda'), torch.tensor(0.0, device='cuda'))
            total = total / len(models)  # len must be an odd number
            results.append(total.tolist())
        return results

    nsml.bind(save=save, load=load, infer=infer)


class Trainer(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH[0])
    UNLABELED_DATA_PATH = '{}/train/raw.json'.format(DATASET_PATH[1])

    def __init__(self, model, model_i, n_models, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, model_i, n_models)
        self.model = model
        self.model.to(self.device)
        self.model_i = model_i
        self.loss_fn = nn.BCELoss()
        self.batch_size = BATCH_SIZE
        self.__test_iter = None
        bind_model(self.model)

    @property
    def test_iter(self) -> Iterator:
        if self.__test_iter:
            self.__test_iter.init_epoch()
            return self.__test_iter
        else:
            self.__test_iter = Iterator(self.task.datasets[1], batch_size=self.batch_size, repeat=False,
                                        sort_key=lambda x: len(x.syllable_contents), train=False,
                                        device=self.device)
            return self.__test_iter

    def train(self):
        max_epoch = 50
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        total_len = len(self.task.datasets[0])
        ds_iter = Iterator(self.task.datasets[0], batch_size=self.batch_size, repeat=False,
                           sort_key=lambda x: len(x.syllable_contents), train=True, device=self.device)
        best_f1_score = 0
        best_state_dict = self.model.state_dict()

        for epoch in range(max_epoch):
            loss_sum, acc_sum = 0., 0.
            ds_iter.init_epoch()
            true_lst = list()
            pred_lst = list()

            self.model.train()
            for i, batch in enumerate(ds_iter):
                self.model.zero_grad()
                pred = self.model(batch.syllable_contents)
                acc = torch.sum((torch.reshape(pred, [-1]) > 0.5) == (batch.eval_reply > 0.5), dtype=torch.float32)
                loss = self.loss_fn(pred, batch.eval_reply)
                loss.backward()
                optimizer.step()

                true_lst += batch.eval_reply.tolist()
                pred_lst += pred.tolist()
                acc_sum += acc.tolist()
                loss_sum += loss.tolist() * len(batch)

            # calc training f1-score
            y_true = np.array(true_lst) > 0.5
            y_pred = np.array(pred_lst) > 0.5
            train_recall_score = recall_score(y_true, y_pred)
            train_precision_score = precision_score(y_true, y_pred)
            train_f1_score = f1_score(y_true, y_pred)
            # print(json.dumps(
            #     {'type': 'train', 'dataset': 'hate_speech',
            #      'epoch': epoch, 'f1': train_f1_score, 'loss': loss_sum / total_len, 'acc': acc_sum / total_len,
            #      'recall': train_recall_score, 'precision': train_precision_score}))

            true_lst, pred_lst, loss_avg, acc_lst, te_total = self.eval(self.test_iter, len(self.task.datasets[1]))

            # calc test f1-score
            y_true = np.array(true_lst) > 0.5
            y_pred = np.array(pred_lst) > 0.5
            test_recall_score = recall_score(y_true, y_pred)
            test_precision_score = precision_score(y_true, y_pred)
            test_f1_score = f1_score(y_true, y_pred)
            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'epoch': epoch, 'f1': test_f1_score, 'loss': loss_avg,  'acc': sum(acc_lst) / te_total,
                 'recall': test_recall_score, 'precision': test_precision_score}))
            nsml.save(str(self.model_i) + '-' + str(epoch))
            if test_f1_score > best_f1_score:
                best_state_dict = model.state_dict()
                nsml.save(str(self.model_i) + '-best')
                best_f1_score = test_f1_score

            # plot graphs
            test_loss = loss_avg
            train_loss = loss_sum / total_len
            nsml.report(step=epoch, test_f1=test_f1_score, test_loss=test_loss,
                        test_recall=test_recall_score, test_precision=test_precision_score)
            nsml.report(step=epoch, train_f1=train_f1_score, train_loss=train_loss,
                        train_recall=train_recall_score, train_precision=train_precision_score)

        return best_state_dict, best_f1_score

    def eval(self, iter:Iterator, total:int) -> (List[float], float, List[float], int):
        true_lst = list()
        pred_lst = list()
        acc_lst = list()
        loss_sum = 0.

        self.model.eval()
        for i, batch in enumerate(iter):
            preds = self.model(batch.syllable_contents)
            accs = torch.eq(preds > 0.5, batch.eval_reply > 0.5).to(torch.float)
            losses = self.loss_fn(preds, batch.eval_reply)
            true_lst += batch.eval_reply.tolist()  # real label
            pred_lst += preds.tolist()  # prediction
            acc_lst += accs.tolist()
            loss_sum += losses.tolist() * len(batch)

        return true_lst, pred_lst, loss_sum / total, acc_lst, total

    def save_model(self, model, appendix=None):
        file_name = 'model'
        if appendix:
            file_name += '_{}'.format(appendix)
        torch.save({'model': model, 'task': type(self.task).__name__}, file_name)


if __name__ == '__main__':
    # Constants
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.3
    EMBEDDING_SIZE = 128
    BATCH_SIZE = 512
    BI_RNN_LAYERS = 1
    UNI_RNN_LAYERS = 1
    LEARNING_RATE = 0.001
    ENSEMBLE_SIZE = 5

    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pause', default=0)
    args = parser.parse_args()
    task = HateSpeech()
    vocab_size = task.max_vocab_indexes['syllable_contents']
    models = []
    for i in range(ENSEMBLE_SIZE):
        model = BaseLine(HIDDEN_DIM, DROPOUT_RATE, vocab_size, EMBEDDING_SIZE, BI_RNN_LAYERS, UNI_RNN_LAYERS)
        model.to('cuda')
        models.append(model)

    if args.pause:
        bind_models(models)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        scores = []
        for i, model in enumerate(models):
            trainer = Trainer(model, i, ENSEMBLE_SIZE, device='cuda')
            best_state_dict, best_f1_score = trainer.train()
            model.load_state_dict(best_state_dict)
            scores.append(best_f1_score)
            print('best f1 score:', best_f1_score)

        print('avg f1 score:', sum(scores) / len(scores))

        bind_models(models)
        nsml.save('ensemble')
