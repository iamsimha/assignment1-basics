import hydra
import numpy as np
import os
import time
import mlflow
import gc
import json
from omegaconf import DictConfig, OmegaConf
from mlflow_logging import _setup_mlflow, _teardown_mlflow
from typing import Optional, Any, Mapping
from tqdm.auto import tqdm
from gpu_picker import get_gpu
from nn_serialization import save_checkpoint

def save_model_config(cfg, dir):
    fname = os.path.join(dir, "config.json")
    with open(fname, "w") as fp:
        json.dump(
            {"d_model": cfg.d_model,
            "num_layers": cfg.n_layers,
            "num_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "max_seq_len": cfg.context_length,
            "theta": cfg.theta,
            "vocab_size": cfg.vocab_size,
            "device": "cuda",
            "pre_norm": cfg.pre_norm,
            "use_rope": cfg.use_rope
            }, fp, indent=2)

def save_optimizer_config(cfg, dir):
    fname = os.path.join(dir, "optim_config.json")
    with open(fname, "w") as fp:
        json.dump({"lr": cfg.lr}, fp, indent=2)


def load_tranformer(cfg):
    from nn_transformer import Transformer
    transformer = Transformer(d_model=cfg.d_model,
                            num_layers=cfg.n_layers,
                            num_heads=cfg.n_heads,
                            d_ff=cfg.d_ff,
                            max_seq_len=cfg.context_length,
                            theta=cfg.theta,
                            vocab_size=cfg.vocab_size,
                            device="cuda",
                            pre_norm=cfg.pre_norm,
                            use_rope=cfg.use_rope)
    return transformer

def get_dataset(dataset_path):
    file_size = os.path.getsize(dataset_path)
    count = file_size // np.dtype(np.uint16).itemsize
    arr = np.memmap(dataset_path, dtype=np.uint16, mode="r", shape=(count,))
    return arr

def compute_validation_loss(model, dataset, batch_size, context_length):
    import torch
    from nn_data import get_batch_seq
    from nn_loss import cross_entropy

    val_dataset = get_batch_seq(dataset, batch_size,
                        context_length, "cuda")
    total_nll = 0
    total_toks = 0
    model.eval()
    for batch in tqdm(val_dataset):
        inputs, targets = batch
        with torch.no_grad():
            logits = model(inputs)
            loss = cross_entropy(logits, targets, reduction="sum")
            total_nll += loss.item()
            total_toks += targets.numel()
    avg_loss = total_nll/total_toks
    avg_perp = np.exp(avg_loss)
    model.train()
    return avg_loss, avg_perp


def train(cfg):
    import torch
    from nn_data import get_batch, get_batch_seq
    from nn_optim import AdamW, SingleDeviceMuon
    from nn_loss import cross_entropy
    from nn_serialization import save_checkpoint
    gid, _lock = get_gpu(threshold_mb=100, sleep_seconds=10)
    print(f"[GPU-PICK] Assigned GPU {gid}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    torch.cuda.set_device(0)

    try:
        # _setup_mlflow(cfg)
        checkpoint_dir = os.path.join(cfg.checkpoint_path, f"lr_{cfg.lr}", f"b_{cfg.train_batch_size}")
        model = load_tranformer(cfg.transformer_arch_params)
        save_model_config(cfg.transformer_arch_params, checkpoint_dir)
        save_optimizer_config(cfg, checkpoint_dir)
        exit()
        train_dataset = get_dataset(cfg.train_dataset_path)
        val_dataset = get_dataset(cfg.validation_dataset_path)
        num_steps = cfg.max_training_tokens // (cfg.train_batch_size * cfg.transformer_arch_params.context_length)
        if cfg.optimizer == "AdamW":
            optimizer = AdamW(model.parameters(), cfg.lr)
        elif cfg.optimizer == "Muon":
            optimizer = SingleDeviceMuon(model.parameters(), lr=cfg.lr)

        tokens_per_step = cfg.train_batch_size * cfg.transformer_arch_params.context_length
        mlflow.log_param("tokens_per_step", int(tokens_per_step))
        mlflow.log_param("num_steps", int(num_steps))
        mlflow.set_tag("device", gid)
        t0 = time.time()
        processed_tokens = 0
        for step in tqdm(range(1, num_steps + 1)):

            inputs, targets = get_batch(train_dataset, cfg.train_batch_size,
                                cfg.transformer_arch_params.context_length, "cuda")
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()


            # ML flow logging
            processed_tokens += tokens_per_step
            elapsed_time = time.time() - t0
            tps = processed_tokens / elapsed_time

            mlflow.log_metric("loss", float(loss.item()), step=step)
            mlflow.log_metric("tokens_per_sec", float(tps), step=step)
            mlflow.log_metric("processed_tokens", int(processed_tokens), step=step)


            # compute validation loss
            if step > 0 and step % cfg.log_every == 0:
                val_loss, val_perp = compute_validation_loss(model,
                        val_dataset, cfg.val_batch_size, cfg.transformer_arch_params.context_length)

                mlflow.log_metric("val_loss", val_loss, step=step)

                mlflow.log_metric("val_perp", val_perp, step=step)

            # Save model

            if step > 0 and step % cfg.save_every == 0:
                # save checkpoint

                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "model.bin")
                save_checkpoint(model, optimizer, step, checkpoint_path)
                print(f"checkpoint saved at {checkpoint_path} during step: {step}")

        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        _teardown_mlflow()

@hydra.main(config_path="config", config_name="arch_ablation.yaml", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    train(cfg)
    print("*"*32)
    print("Training end")

if __name__ == "__main__":
    main()