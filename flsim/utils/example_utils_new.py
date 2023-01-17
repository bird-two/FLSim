#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# utils for use in the examples and tutorials

import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder, SequentialSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.data.data_utils import batchify
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from flsim.data.partition import SubDataset, build_distribution
from torch import nn
from torch.utils.data import DataLoader



def collate_fn(batch: Tuple) -> Dict[str, Any]:
    feature, label = batch
    return {"features": feature, "labels": label}


class UserData(IFLUserData):
    def __init__(self, user_data: Generator, eval_flag: bool = False):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0

        if eval_flag == False:
            self._train_batches= list(user_data)
        else:
            self._eval_batches = list(user_data)

        self._num_train_batches = len(self._train_batches)
        self._num_eval_batches = len(self._eval_batches)
        # for features, labels in zip(user_features, user_labels):
        #     if self._num_eval_examples < int(total * self._eval_split):
        #         self._num_eval_batches += 1
        #         self._num_eval_examples += UserData.get_num_examples(labels)
        #         self._eval_batches.append(UserData.fl_training_batch(features, labels))
        #     else:
        #         self._num_train_batches += 1
        #         self._num_train_examples += UserData.get_num_examples(labels)
        #         self._train_batches.append(UserData.fl_training_batch(features, labels))

    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """
        return self._num_train_examples

    def num_eval_examples(self):
        """
        Returns the number of eval examples
        """
        return self._num_eval_examples

    def num_train_batches(self):
        """
        Returns the number of train batches
        """
        return self._num_train_batches

    def num_eval_batches(self):
        """
        Returns the number of eval batches
        """
        return self._num_eval_batches

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data for training
        """
        for batch in self._train_batches:
            features, labels = batch 
            yield {"features": features, "labels": labels}

    def eval_data(self):
        """
        Iterator to return a user batch data for evaluation
        """
        for batch in self._eval_batches:
            features, labels = batch 
            yield {"features": features, "labels": labels}

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    # @staticmethod
    # def fl_training_batch(
    #     features: List[torch.Tensor], labels: List[float]
    # ) -> Dict[str, torch.Tensor]:
    #     return {"features": torch.stack(features), "labels": torch.Tensor(labels)}

class DataProvider(IFLDataProvider):
    def __init__(self, train_dataset, test_dataset, net_dataidx_map, local_batch_size):
        self.train_loaders = {}
        for user_index in range(len(net_dataidx_map)):
            train_loader = DataLoader(
                SubDataset(train_dataset, net_dataidx_map[user_index]), 
                batch_size=local_batch_size, 
                shuffle=True,
                )
            self.train_loaders[user_index] = UserData(user_data=train_loader, eval_flag=False)
            #self.train_loaders[user_index] = train_loader
        #bd Note [2023 01 13] The definitions of evaluate and test are ambiguous
        self.eval_loader = DataLoader(test_dataset, batch_size=local_batch_size, shuffle=False)
        self.eval_loaders = {0: UserData(user_data=self.eval_loader, eval_flag=True)}

    def train_user_ids(self) -> List[int]:
        return list(self.train_loaders.keys())

    def num_train_users(self) -> int:
        return len(self.train_loaders)

    def get_train_user(self, user_index: int) -> IFLUserData:
        r'''Get the `IFLUserData` by the user_index'''
        if user_index in self.train_loaders:
            return self.train_loaders[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self.train_loaders.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self.eval_loaders.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        yield from self.eval_users()


class FLModel(IFLModel):
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def fl_forward(self, batch) -> FLBatchMetrics:
        features = batch["features"]  # [B, C, 28, 28]
        batch_label = batch["labels"]
        stacked_label = batch_label.view(-1).long().clone().detach()
        if self.device is not None:
            features = features.to(self.device)

        output = self.model(features)

        if self.device is not None:
            output, batch_label, stacked_label = (
                output.to(self.device),
                batch_label.to(self.device),
                stacked_label.to(self.device),
            )

        loss = F.cross_entropy(output, stacked_label)
        num_examples = self.get_num_examples(batch)
        output = output.detach().cpu()
        stacked_label = stacked_label.detach().cpu()
        del features
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=stacked_label,
            model_inputs=[],
        )

    def fl_create_training_batch(self, **kwargs):
        features = kwargs.get("features", None)
        labels = kwargs.get("labels", None)
        return UserData.fl_training_batch(features, labels)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)  # pyre-ignore

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return UserData.get_num_examples(batch["labels"])


class MetricsReporter(FLMetricsReporter):
    ACCURACY = "Accuracy"

    def __init__(
        self,
        channels: List[Channel],
        target_eval: float = 0.0,
        window_size: int = 5,
        average_type: str = "sma",
        log_dir: Optional[str] = None,
    ):
        super().__init__(channels, log_dir)
        self.set_summary_writer(log_dir=log_dir)
        self._round_to_target = float(1e10)

    def compare_metrics(self, eval_metrics: dict, best_metrics: dict):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True

        current_accuracy = eval_metrics.get(self.ACCURACY, float("-inf"))
        best_accuracy = best_metrics.get(self.ACCURACY, float("-inf"))
        return current_accuracy > best_accuracy

    def compute_scores(self) -> Dict[str, Any]:
        # compute accuracy
        correct = torch.Tensor([0])
        for i in range(len(self.predictions_list)):
            all_preds = self.predictions_list[i]
            pred = all_preds.data.max(1, keepdim=True)[1]

            assert pred.device == self.targets_list[i].device, (
                f"Pred and targets moved to different devices: "
                f"pred >> {pred.device} vs. targets >> {self.targets_list[i].device}"
            )
            if i == 0:
                correct = correct.to(pred.device)

            correct += pred.eq(self.targets_list[i].data.view_as(pred)).sum()

        # total number of data
        total = sum(len(batch_targets) for batch_targets in self.targets_list)

        accuracy = 100.0 * correct.item() / total
        return {self.ACCURACY: accuracy}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        accuracy = scores[self.ACCURACY]
        return {self.ACCURACY: accuracy}


if __name__ == '__main__':
    from flsim.data.datasets import build_dataset
    net_dataidx_map = build_distribution(data_path='../flsim/data/partitionfiles/inuse/', dtype='mnist', num_users=10, way='dir-u', alpha=1.0)
    local_batch_size = 100
    train_dataset, test_dataset =  build_dataset('mnist', data_path='../data/')
    provider = DataProvider(train_dataset, test_dataset, net_dataidx_map, local_batch_size)
    for batch in provider.train_loaders[0]:
        print(batch)
        break

