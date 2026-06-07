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



def make_stacked_moons(split_name, n_per_global, storage_dir_dataset, n_global=3, noise=0.05, vertical_shift=3.0):
    """
    Creates a hierarchical dataset from stacked 'two moons'.
    Each moon level is a global cluster.
    Upper and lower halves represent local clusters.
    """
    X_list, global_labels_list, local_labels_list = [], [], []
    
    for g in range(n_global):
        X, y = datasets.make_moons(n_samples=n_per_global, noise=noise)
        
        # Vertical shifts for each global cluster
        if g == 1:
            X[:, 1] += vertical_shift
        elif g == 2:
            X[:, 0] += vertical_shift
            X[:, 1] += vertical_shift / 2
        
        X_list.append(X)
        global_labels_ = [g] * n_per_global
        global_labels_list.extend(torch.nn.functional.one_hot(
            torch.tensor(global_labels_).long(), num_classes=n_global
        ).float())

        local_labels_list.extend(y if g != 2 else np.abs(y - 1))
    
    X = np.vstack(X_list)
    global_labels = np.array(global_labels_list)
    local_labels = np.array(local_labels_list)

    storage_dir_dataset = Path(storage_dir_dataset) / split_name
    storage_dir_dataset.mkdir(parents=True, exist_ok=True)
    
    dataset_dict = {}
    for idx in range(len(X)):
        example_id = f"{idx}"
        np.save(storage_dir_dataset / f"{example_id}.npy", X[idx])
        dataset_dict[example_id] = {
            "observation": str(storage_dir_dataset / f"{example_id}.npy"),
            "high_level": global_labels[idx].tolist(),
            "low_level": local_labels[idx].tolist(),
        }
    return dataset_dict


def make_moons(split_name, num_examples, storage_dir_dataset):
    X, y = datasets.make_moons(n_samples=num_examples, noise=.05)
    storage_dir = Path(storage_dir_dataset) / split_name
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dict = {}
    for idx in range(num_examples):
        example_id = f"{idx}_{split_name}"
        np.save(storage_dir / f"{example_id}.npy", X[idx])
        dataset_dict[example_id] = {
            "observation": str(storage_dir / f"{example_id}.npy"),
            "speaker_conditioning": int(y[idx]),
        }
    return dataset_dict


def prepare_example(example):
    observation = np.load(example['observation'])
    example['observation'] = observation.tolist()
    return example


def get_dataset(
        batch_size, 
        storage_dir, 
        dataset_name="moons",
        buffer_size=5 * 10000
    ):
    storage_dir = Path(storage_dir) / dataset_name
    dataset_file = storage_dir / "dataset.json"

    if not dataset_file.exists():
        if dataset_name == "stacked_moons":
            dataset_fkt = make_stacked_moons
        elif dataset_name == "moons":            
            dataset_fkt = make_moons
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        storage_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict = dict(
            train=dataset_fkt("train", 5000, storage_dir),
            eval=dataset_fkt("eval", 1000, storage_dir),
            test=dataset_fkt("test", 1000, storage_dir),
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
    cfg.trainer.storage_dir = Path(cfg.trainer.storage_dir) / cfg.ex_name
    cfg.trainer.storage_dir.mkdir(parents=True, exist_ok=True)
    pb.io.dump_yaml(
        OmegaConf.to_container(cfg, resolve=True),
        cfg.trainer.storage_dir / "config.yaml"
    )

    trainer = Trainer.get_config(
        OmegaConf.to_container(cfg.trainer, resolve=True)
    )
    trainer = Trainer.from_config(trainer)

    train, eval = get_dataset(
        cfg.batch_size, 
        cfg.storage_dir_dataset, 
        dataset_name=cfg.dataset_name
    )

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
