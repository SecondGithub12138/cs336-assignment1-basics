import argparse
import time
import json
import os
import torch
import numpy as np
import wandb
from einops import rearrange
from pathlib import Path

from tests.adapters import (
    TransformerLM,
    AdamW,
    get_batch,
    cross_entropy,
    lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
)

EVAL_BATCH_NUM = 10

def main():
    parser = argparse.ArgumentParser(description="Train a transformer language model")

    # Model argument 
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--rope_theta", type=float)
    # weights: dict[str, Tensor],
    # in_indices: Int[Tensor, " batch_size sequence_length"],

    # Optimizer argument 
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--weight_decay", type=float)
    # params

    # Training arguments
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_l2_norm", type=float)
    # *lr_cosine_schedule
    parser.add_argument("--max_learning_rate", type=float)
    parser.add_argument("--min_learning_rate", type=float)
    parser.add_argument("--warmup_iters", type=int)
    parser.add_argument("--cosine_cycle_iters", type=int)
    parser.add_argument("--max_steps", type=int)
    # *checkpoint
    parser.add_argument("--checkpoint_interval", type=int)

    # Data arguments
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--resume", type=str)

    # Others
    parser.add_argument("--device", type=str)
    parser.add_argument("--config", type=str)
    # parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--log_interval", type=int)

    args = parser.parse_args()

    # 覆盖config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        with open("config_default.json", 'r') as f:
            config = json.load(f)

    # Override with cli flag
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    
    if config["use_wandb"]:
        wandb_run = wandb.init(
            project="CS336",
            name="CZF training",
            config=config
        )
    # 然后再看看怎么拼起来吧
    # 第一步先把文件load进来 training 
    print("Loading datasets...")
    train_data = load_memmap_data(config["train_data"])
    val_data = load_memmap_data(config["val_data"])

    # 接下来开始训练，
    # 首先一步load weights，最开始没有weight呀怎么办
    # 别急，先构建大框架，一共要训练多少step，每个step里都要做什么？
    # 结果发现没有构建query 和 comparing， 去找train 和 val data吧
    
    step = 1
    transformer_model = TransformerLM(config["vocab_size"], config["context_length"], config["d_model"], config["num_layers"], config["num_heads"], config["d_ff"], config["rope_theta"], None)
    # make sure everything under "cuda"!wget https://huggingface.co/datasets/chenwudi12138/owt.bin/resolve/main/owt_valid.bin
    transformer_model = transformer_model.to(config["device"])
    adamw = AdamW(transformer_model.parameters(), config["lr"], (config["beta1"], config["beta2"]), config["eps"], config["weight_decay"])
    if config.get("resume"):
        resume_dir = config.get("resume")
        print(f"read checkpoint {resume_dir}")
        step = run_load_checkpoint(config["resume"], transformer_model, adamw)

    batch_start_time = time.time()  # Track time for every 100 steps
    while step <= config["max_steps"]:
        start_time = time.time()
        input, label = get_batch(train_data, config["batch_size"], config["context_length"], config["device"])
        # 1.前向传播 run_transformer_lm
        # in_indices: Int[Tensor, " batch_size sequence_length"]
        logit = transformer_model(input)
        # 2.计算loss run_cross_entropy
        # preprocess to flat 
        logit = rearrange(logit, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
        label = rearrange(label, "batch_size context_length -> (batch_size context_length)")
        loss = cross_entropy(logit, label)
        # 3.反向传播 loss.backward()
        adamw.zero_grad()
        loss.backward()
        run_gradient_clipping(transformer_model.parameters(), max_l2_norm=config["max_l2_norm"])
        # 4.优化器更新权重 optimizer.step() optimizer.zero_grad()
        lr = lr_cosine_schedule(step, config["max_learning_rate"], config["min_learning_rate"], config["warmup_iters"], config["cosine_cycle_iters"])
        for param_groups in adamw.param_groups:
            param_groups["lr"] = lr
        adamw.step()
        # logging | wandb
        if config["use_wandb"] and step % config["log_interval"] == 0:
            val_loss = eval(val_data, config, transformer_model, EVAL_BATCH_NUM)
            train_loss = eval(train_data, config, transformer_model, EVAL_BATCH_NUM)

            log_data = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "step": step,
                "step_time": time.time() - start_time,
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
            }

            # Print to terminal
            print(f"[Step {step}] loss={loss.item():.4f}, lr={lr:.6f}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={time.time()-start_time:.2f}s")

            # Log to wandb
            wandb_run.log(log_data)

        if step % config["checkpoint_interval"] == 0:
            # 留个心眼看看这个到底是checkpoints/step_1是存在了哪个目录下，应该无论相对本py还是项目根目录都存在assignment1*下.. 看不出来
            checkpoint_dir = Path(config["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            run_save_checkpoint(transformer_model, adamw, step, f"{checkpoint_dir}/step_{step}" )
        if step % 100 == 0:
            batch_time = time.time() - batch_start_time
            print(f"[Step {step}] loss={loss.item():.4f}, lr={lr:.6f}, time={batch_time:.2f}s (100 steps)")
            batch_start_time = time.time()  # Reset timer for next 100 steps
        step+=1

    # Save final checkpoint after training loop
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_save_checkpoint(transformer_model, adamw, step, f"{checkpoint_dir}/step_final" )
    # 还有eval 用上val_data

@torch.no_grad()
def eval(dataset: np.ndarray, config: dict, transformer_model: torch.nn.Module, eval_batch_num: int):
    losses = []
    for _ in range(eval_batch_num):
        batch = get_batch(dataset, config["batch_size"], config["context_length"], config["device"])
        # 1.前向传播 run_transformer_lm
        # in_indices: Int[Tensor, " batch_size sequence_length"]
        logit = transformer_model(batch[0])
        # 2.计算loss run_cross_entropy
        # preprocess to flat 
        logit = rearrange(logit, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
        target = rearrange(batch[1], "batch_size context_length -> (batch_size context_length)")
        loss = cross_entropy(logit, target)
        losses.append(loss.item())
    return np.mean(losses)



def load_memmap_data(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.memmap(path, dtype=np.int32, mode='r')
    print(f"load {len(data):,} tokens from {path}")
    return data

if __name__ == "__main__":
    main()
