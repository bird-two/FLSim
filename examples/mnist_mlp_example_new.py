#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train an image classifier with FLSim to simulate a federated learning training environment.

With this tutorial, you will learn the following key components of FLSim:
1. Data loading
2. Model construction
3. Trainer construction

    Typical usage example:
    python3 mnist_mlp_example_new.py --config-file configs/mnist_mlp_config.json
"""
import flsim.configs  # noqa
import hydra
import torch
import flsim
from flsim.data.data_sharder import SequentialSharder
from flsim.data.datasets import build_dataset
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils_new import (
    DataProvider,
    FLModel,
    MetricsReporter,
)
from flsim.data.partition import build_distribution
from flsim.models.mlp import MLP
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


IMAGE_SIZE = 32


# def build_data_provider(local_batch_size, examples_per_user, drop_last: bool = False):
#     train_dataset, test_dataset = build_dataset(dtype='mnist')
#     sharder = SequentialSharder(examples_per_shard=examples_per_user)
#     fl_data_loader = DataLoader(
#         train_dataset, test_dataset, test_dataset, sharder, local_batch_size, drop_last
#     )
#     data_provider = DataProvider(fl_data_loader)

#     print(f"Clients in total: {data_provider.num_train_users()}")
#     return data_provider


def main(
    trainer_config,
    data_config,
    use_cuda_if_available: bool = True,
) -> None:
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    #model = SimpleConvNet(in_channels=3, num_classes=10)
    model = MLP(dim_in=28*28, dim_hidden=64, dim_out=10)
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")
    
    train_dataset, test_dataset = build_dataset(dtype='mnist', data_path='../data/')
    net_dataidx_map = build_distribution(
        data_path='../flsim/data/partitionfiles/inuse/', 
        dtype='mnist', num_users=10, way='dir-u', alpha=1.0
        )
    data_provider = DataProvider(train_dataset, test_dataset, net_dataidx_map, data_config.local_batch_size)

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="cifar10_tutorial")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data

    main(
        trainer_config,
        data_config,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)