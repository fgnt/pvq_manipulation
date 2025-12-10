import hydra
import lazy_dataset
import numpy as np
import paderbox as pb
import padertorch as pt
import torch

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from padertorch import Trainer
from padertorch.train.hooks import LRSchedulerHook
from sklearn import datasets


def make_split(split_name, num_examples, storage_dir_dataset):
    x, y = datasets.make_moons(n_samples=num_examples, noise=.05)
    storage_dir_dataset_sub = storage_dir_dataset / split_name
    storage_dir_dataset_sub.mkdir(parents=True, exist_ok=True)

    dataset_dict = {}
    for idx in range(num_examples):
        example_id = f"{idx}_{split_name}"
        np.save(storage_dir_dataset_sub / f"{example_id}.npy", x[idx])
        dataset_dict[example_id] = {
            "observation": storage_dir_dataset_sub / f"{example_id}.npy",
            "label": y[idx].tolist(),
        }
    return dataset_dict


def prepare_example(example):
    observation = np.load(example['observation'])
    example['observation'] = observation[None, :].tolist()
    example['speaker_conditioning'] = example['label']
    return example


def get_dataset(batch_size, storage_dir, buffer_size=50):
    storage_dir = Path(storage_dir)
    dataset_file = storage_dir / "dataset.json"

    if not dataset_file.exists():
        storage_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict = dict(
            train=make_split("train", 5000, storage_dir),
            eval=make_split("eval", 1000, storage_dir),
            test=make_split("test", 1000, storage_dir),
        )
        pb.io.dump_json(dataset_dict, dataset_file)
    else:
        dataset_dict = pb.io.load_json(dataset_file)

    def build(split):
        ds = lazy_dataset.from_dict(dataset_dict[split])
        ds = ds.map(prepare_example)
        print("Loading datasets in cache")
        ds = ds.cache(lazy=False)
        return ds.shuffle(reshuffle=True, buffer_size=buffer_size).batch(batch_size, drop_last=True)
    return build("train"), build("eval")


@hydra.main(version_base=None, config_path="configs", config_name="config_toy_example")
def main(cfg: DictConfig):
    cfg.trainer.storage_dir = pt.io.get_new_subdir(
        cfg.trainer.storage_dir / cfg.ex_name,
        id_naming='index',
        mkdir=True,
    )
    cfg.trainer.storage_dir.mkdir(parents=True, exist_ok=True)
    pb.io.dump_yaml(
        OmegaConf.to_container(cfg, resolve=True),
        cfg.trainer.storage_dir / "config.yaml"
    )

    trainer = Trainer.get_config(OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer = Trainer.from_config(trainer)

    train, eval = get_dataset(cfg.batch_size, cfg.storage_dir_dataset)

    trainer.register_validation_hook(eval, early_stopping_patience=10)
    trainer.register_hook(
        LRSchedulerHook(
            torch.optim.lr_scheduler.StepLR(
                trainer.optimizer.optimizer,
                step_size=100,
                gamma=0.98
            )
        )
    )
    trainer.test_run(train, eval)
    trainer.train(train)


if __name__ == "__main__":
    main()
