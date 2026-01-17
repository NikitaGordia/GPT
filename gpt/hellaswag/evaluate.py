import json
from typing import Any, Callable, Dict, Generator, List, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig
import tiktoken
from tiktoken import Encoding
import torch
import torch.distributed as dist
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from gpt.utils import RuntimeEnvironment, setup_env

from .dataset import prepare_dataset


def iterate_examples(
    cfg: DictConfig, env: RuntimeEnvironment
) -> Generator[Dict[str, Any], None, None]:
    """Iterate through examples in a specific split of the HellaSwag dataset.

    Args:
        cfg: Hydra configuration
        env: Runtime environment with device and configuration settings
    Yields:
        Dictionary containing a single example from the dataset
    """
    fname = prepare_dataset(cfg, env)
    with open(fname, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def render_example(
    example: str, enc: Encoding
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, int]:
    """Process a HellaSwag example into tensors for model evaluation.

    Args:
        example: A single example from the HellaSwag dataset
        enc: Tokenizer encoding to use for tokenization

    Returns:
        Tuple containing:
            - Processed data dictionary with tokens
            - Tensor of token IDs for context and endings
            - Tensor of masks indicating which tokens are from endings
            - Integer label indicating the correct ending
    """
    ctx = example["ctx"]
    label = int(example["label"])
    endings = example["endings"]

    data: Dict[str, Any] = {"label": label, "ctx_tokens": enc.encode(ctx), "ending_tokens": []}

    tok_rows: List[List[int]] = []
    mask_rows: List[List[int]] = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(data["ctx_tokens"] + end_tokens)
        mask_rows.append([0] * len(data["ctx_tokens"]) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max([len(row) for row in tok_rows])
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


@torch.no_grad()
def evaluate(
    cfg_hellaswag: DictConfig,
    env: RuntimeEnvironment,
    model: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    encoder: Encoding,
) -> Tuple[int, int, int]:
    """Evaluate a model on the HellaSwag validation dataset.

    Args:
        cfg: Hydra configuration
        env: Training environment with device and configuration settings
        model: Model function that takes tokens and returns (logits, _)
        encoder: Tokenizer encoding to use for tokenization

    Returns:
        Tuple containing:
            - Total number of examples evaluated
            - Number of correct predictions using sum loss
            - Number of correct predictions using normalized loss
    """

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    ddp_rank, ddp_world_size = env.ddp_config

    for ix, example in tqdm(enumerate(iterate_examples(cfg_hellaswag, env))):
        if ix % ddp_world_size != ddp_rank:
            continue

        _, tokens, mask, label = render_example(example, encoder)
        tokens, mask = tokens.to(env.device), mask.to(env.device)

        if env.use_autocast:
            with torch.autocast(device_type=env.device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
        else:
            logits, _ = model(tokens)

        # Shift logits and tokens for next-token prediction
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()

        # Flatten for cross entropy calculation
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        # Calculate token-wise losses
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Apply mask to only consider losses on ending tokens
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask

        # Calculate sum and average losses for each ending
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Get predictions (lowest loss is best)
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

    # Aggregate results across distributed processes if using DDP
    if env.ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=env.device)
        num_correct = torch.tensor(num_correct, dtype=torch.long, device=env.device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=env.device)

        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)

        num_total = num_total.item()
        num_correct = num_correct.item()
        num_correct_norm = num_correct_norm.item()

    return num_total, num_correct, num_correct_norm


def render_results(num_total: int, num_correct: int, num_correct_norm: int) -> str:
    return (
        (
            f"Total examples: {num_total}, Accuracy: {num_correct / num_total * 100:.1f}%, "
            f"NormAccuracy: {num_correct_norm / num_total * 100:.1f}%"
        )
        if num_total > 0
        else "No examples evaluated"
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def evaluate_pretrained(cfg: DictConfig) -> None:
    """
    Evaluate a pretrained model on the HellaSwag dataset.

    Args:
        cfg: Hydra configuration
    """

    env = setup_env(cfg.env)

    hs_cfg = cfg.data.hellaswag

    model = GPT2LMHeadModel.from_pretrained(cfg.model.name)
    model.to(env.device)

    encoder = tiktoken.get_encoding(cfg.tokens.encoding)

    def model_fn(x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return model(x).logits, None

    num_total, num_correct, num_correct_norm = evaluate(hs_cfg, env, model_fn, encoder)
    logger.success(render_results(num_total, num_correct, num_correct_norm))


if __name__ == "__main__":
    evaluate_pretrained()
