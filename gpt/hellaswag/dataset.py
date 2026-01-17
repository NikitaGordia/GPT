import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import requests
import torch.distributed as dist
from tqdm import tqdm

from gpt.utils import RuntimeEnvironment, setup_env


def download_file(url: str, fname: str, chunk_size: int = 1024) -> None:
    """Helper function to download a file from a given url.

    Args:
        url: URL to download from
        fname: Local filename to save to
        chunk_size: Size of chunks to download at a time
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as file,
        tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def prepare_dataset(cfg: DictConfig, env: RuntimeEnvironment, force: bool = False) -> str:
    """Download a specific split of the HellaSwag dataset if not already cached.

    Args:
        cfg: Hydra configuration
        env: Runtime environment with device and configuration settings
        force: Whether to force download even if the file already exists

    Returns:
        Path to the downloaded/cached file
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)

    data_url = cfg.urls[cfg.split]

    data_filename = os.path.join(cfg.cache_dir, f"hellaswag_{cfg.split}.json")
    if (not os.path.exists(data_filename) or force) and env.is_master:
        # env.logger(f"Downloading {data_url} to {data_filename}...") TODO

        download_file(data_url, data_filename, chunk_size=cfg.chunk_size)

    if env.ddp:
        dist.barrier()
    return data_filename


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def cache_dataset(cfg: DictConfig) -> None:
    """
    Download a specific split of the HellaSwag dataset.

    This command can be used to force re-download the dataset files even if they already exist.

    Args:
        cfg: Hydra configuration
    """

    env = setup_env(cfg.env)

    # Validate split
    hs_cfg = cfg.data.hellaswag
    if hs_cfg.split not in cfg.data.hellaswag.urls:
        valid_splits = ", ".join(cfg.data.hellaswag.urls.keys())
        raise ValueError(f"Invalid split: {hs_cfg.split}. Valid splits are: {valid_splits}")

    # Download the file
    data_filename = prepare_dataset(hs_cfg, env, force=True)
    logger.success(f"Downloaded {hs_cfg.split} split to {data_filename}")


if __name__ == "__main__":
    cache_dataset()
