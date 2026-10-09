import comet_ml
from comet_ml.integration.pytorch import log_model, load_model

from run_config import ModelConfig, OptimizerConfig, GeneralConfig
import utils.data_loader as data_loader
from generate import get_prefixes_and_generate
from structures.models import TransformerLM, cross_entropy
from structures.optimizer import AdamW, learning_rate_schedule, gradient_clipping
from utils.io import load_checkpoint, save_checkpoint

import numpy as np
import pathlib
import cProfile, pstats, io
import sys
import time
import torch

# set up comet_ml
comet_ml.login()
PROJECT_NAME = "LLM From Scratch"
WORKSPACE = "sunbri"
EXPERIMENT_NAME = 'tiny'

BASE_PATH = (pathlib.Path(__file__).resolve().parent.parent)
DATA_PATH = BASE_PATH / "data"

INFERENCE_MODE = True

# gets some data and moves it to the right device
def get_batch(file_name: str, batch_size: int, context_length: int, device: str):
    data = np.memmap(DATA_PATH / file_name, dtype=np.uint16, mode='r')
    return data_loader.get_batch(data, batch_size, context_length, device)

if __name__ == "__main__":
    # parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    # i/o and model selection
    # general stuff
    parser.add_argument("--input_train", type=str, default='encoded_TinyStoriesV2-GPT4-train.bin')
    parser.add_argument("--input_val", type=str, default='encoded_TinyStoriesV2-GPT4-valid.bin')
    parser.add_argument("--checkpoint_dir", type=str, default=BASE_PATH / "checkpoints")
    parser.add_argument("--model", type=str, default="tiny")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=400)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    # optimizer
    parser.add_argument("--max_learning_rate", type=float, default=2e-3)
    parser.add_argument("--min_learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.95)
    parser.add_argument("--eps_adam", type=float, default=1e-8)
    parser.add_argument("--max_norm", type=float, default=1.0)
    # loading from checkpoint? and if we do, WE ARE ASSUMING WE ARE
    # CALLING THIS SCRIPT WITH THE SAME HYPERPARAMETERS!!!
    parser.add_argument("--load_from_checkpoint", action='store_true')
    args = parser.parse_args()

    # double check: set CUDA / mps
    if torch.cuda.is_available():
        args.device = "cuda"
    elif torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"
    print(f"Device: {args.device}", flush=True)

    # all of these hyperparameter things are just so that we can
    # have autocomplete in the code editor
    general_hypers = GeneralConfig(
        input_train=args.input_train,
        input_val=args.input_val,
        checkpoint_dir=args.checkpoint_dir,
        model=args.model,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        device=args.device,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        load_from_checkpoint=args.load_from_checkpoint
    )
    model_config = {
        'micro': dict(
            vocab_size = 10000,
            context_length = 100,
            d_model = 64,
            num_layers = 2,
            num_heads = 4,
            d_ff = 256,
            rope_theta = 10000
        ),
        'tiny': dict(
            vocab_size = 10000,
            context_length = 256,
            d_model = 512,
            num_layers = 4,
            num_heads = 16,
            d_ff = 1344,
            rope_theta = 10000
        )
    }[args.model]
    model_hypers = ModelConfig(**model_config)
    optimizer_hypers = OptimizerConfig(
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        weight_decay=args.weight_decay,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        eps_adam=args.eps_adam,
        max_norm=args.max_norm
    )

    # get the model
    model = TransformerLM(
        vocab_size=model_hypers.vocab_size,
        context_length=model_hypers.context_length,
        d_model=model_hypers.d_model,
        num_layers=model_hypers.num_layers,
        num_heads=model_hypers.num_heads,
        d_ff=model_hypers.d_ff,
        rope_theta=model_hypers.rope_theta,
        device=general_hypers.device
    )
    model.to(general_hypers.device)
    print(next(model.parameters()).device)

    # get the optimizer
    opt = AdamW(
        params=model.parameters(),
        weight_decay=optimizer_hypers.weight_decay,
        betas=(optimizer_hypers.beta_1, optimizer_hypers.beta_2),
        eps=optimizer_hypers.eps_adam,
    )

    # this value can be overwritten from a checkpoint
    iter_num = 1

    # checkpoint stuff
    # we are getting the model from our local checkpoint for inference
    checkpoint_path = f"{general_hypers.checkpoint_dir}/{EXPERIMENT_NAME}.pt"
    if general_hypers.load_from_checkpoint:
        # fix this later
        iter_num = load_checkpoint(checkpoint_path, model, opt)

    if INFERENCE_MODE:
        # see download_model.py to see how we download it
        # we need to change map_location since the model itself was saved as a CUDA model
        comet_model = torch.load(f"{BASE_PATH}/models/model-data/comet-torch-model.pth", map_location='mps')
        model.load_state_dict(comet_model)
        get_prefixes_and_generate(model, general_hypers.device)
    else:
        # set up comet_ml for logging
        experiment = comet_ml.start(
            api_key='KEY',
            project_name=PROJECT_NAME,
            workspace=WORKSPACE,
            experiment_config=comet_ml.ExperimentConfig(name=EXPERIMENT_NAME)
        )
        experiment.log_parameters(vars(args))

        with experiment.train():
            # get first training batch
            x, y = get_batch(general_hypers.input_train, general_hypers.batch_size,
                             model_hypers.context_length, general_hypers.device)
            while True:
                opt.zero_grad()
                loss = cross_entropy(model(x), y)

                x, y = get_batch(general_hypers.input_train, general_hypers.batch_size,
                                 model_hypers.context_length, general_hypers.device)

                loss.backward()

                experiment.log_metrics({
                    'loss': loss,
                    'iter_num': iter_num
                })

                gradient_clipping(model.parameters(), optimizer_hypers.max_norm)
                for group in opt.param_groups:
                    group['lr'] = learning_rate_schedule(iter_num,
                                                         optimizer_hypers.max_learning_rate,
                                                         optimizer_hypers.min_learning_rate,
                                                         optimizer_hypers.warmup_iters,
                                                         optimizer_hypers.cosine_cycle_iters)
                opt.step()
                
                if iter_num % general_hypers.eval_interval == 0:
                    total_val_loss = 0
                    for _ in range(general_hypers.eval_iters):
                        x_val, y_val = get_batch(general_hypers.input_val, general_hypers.batch_size,
                                                 model_hypers.context_length, general_hypers.device)
                        with torch.no_grad():
                            iter_val_loss = cross_entropy(model(x_val), y_val)
                            total_val_loss += iter_val_loss.item()
                    val_loss = total_val_loss / general_hypers.eval_iters
                    experiment.log_metric('val_loss', val_loss)

                # if iter_num % 10000 == 0:
                #     model_checkpoint = {
                #         "model_state_dict": model.state_dict(),
                #         "optimizer_state_dict": opt.state_dict(),
                #         "iter_num": iter_num,
                #     }
                #     log_model(experiment, model_checkpoint, f"Model_{EXPERIMENT_NAME}")
                #     save_checkpoint(model, opt, iter_num, checkpoint_path)

                iter_num += 1
                if iter_num > general_hypers.max_iters:
                    break

        log_model(experiment, model, model_name=f"Model_{EXPERIMENT_NAME}")
        experiment.register_model(f"Model_{EXPERIMENT_NAME}")
        experiment.end()
