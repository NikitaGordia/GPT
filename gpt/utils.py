from dataclasses import dataclass
import os
import sys
import time
from typing import Tuple

from loguru import logger
from omegaconf import DictConfig
import torch
from torch.distributed import init_process_group


class Timeit:
    def __init__(self, use_cuda: bool = False) -> None:
        """
        Initialize the context manager for time measurement.

        Args:
            use_cuda (bool, optional): Whether to synchronize CUDA operations for accurate timing.
                                      Defaults to False.
        """
        self.use_cuda = use_cuda

    def __enter__(self):
        """
        Start the timer when entering the context.

        Returns:
            self: The context manager instance.
        """
        self.start_time = time.time()

        if self.use_cuda:
            torch.cuda.synchronize()

        return self

    def __exit__(self, *_) -> bool:
        return False

    def now(self) -> float:
        # Handle PyTorch CUDA timing if available
        if self.use_cuda:
            torch.cuda.synchronize()

        # Standard Python timing
        elapsed_time = time.time() - self.start_time

        # The context manager doesn't suppress exceptions
        return elapsed_time


@dataclass
class RuntimeEnvironment:
    use_autocast: bool = False
    use_tf32: bool = False
    distributed_backend: str = "nccl"

    ddp: bool = False
    ddp_config: Tuple[int, int] = (0, 1)
    ddp_local_rank: int = 0
    device: str = None

    @property
    def ddp_rank(self) -> bool:
        return self.ddp_config[0]

    @property
    def ddp_world_size(self) -> bool:
        return self.ddp_config[1]

    @property
    def is_master(self) -> bool:
        return self.ddp_rank == 0

    @property
    def device_type(self) -> str:
        return "cuda" if self.device.startswith("cuda") else "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def use_cpu(self) -> bool:
        return self.device == "cpu"

    def setup_accelerators(self, device_mode) -> None:
        ddp = int(os.environ.get("RANK", -1)) != -1
        if ddp:
            assert torch.cuda.is_available(), "DDP requires CUDA device"
            init_process_group(backend=self.distributed_backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            ddp_config = (ddp_rank, ddp_world_size)
            ddp = True
        else:
            ddp_config = (0, 1)
            ddp_local_rank = 0
            device = detect_device(device_mode)
            ddp = False

        self.ddp_config = ddp_config
        self.ddp_local_rank = ddp_local_rank
        self.device = device
        self.ddp = ddp

    @staticmethod
    def from_hydra_config(cfg: DictConfig):
        env = RuntimeEnvironment(
            use_autocast=cfg.use_autocast,
            use_tf32=cfg.use_tf32,
            distributed_backend=cfg.distributed_backend,
        )
        env.setup_accelerators(cfg.device_mode)
        return env


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detect_device(mode: str) -> str:
    if mode == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        return device
    else:
        return mode


def configure_logger(rank, log_dir="logs/loguru"):
    logger.remove()

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"rank_{rank}.log")

    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
        enqueue=True,
        diagnose=True,
    )

    if rank == 0:
        logger.add(
            sys.stderr,
            level="DEBUG",
            colorize=True,
            enqueue=True,
            diagnose=False,
        )

    logger.info(f"Logger configured for rank {rank}. Logging to {log_file_path}")


def setup_env(cfg_env: DictConfig) -> RuntimeEnvironment:
    env = RuntimeEnvironment.from_hydra_config(cfg_env)
    configure_logger(env.ddp_rank)

    logger.info("------------------------ ENVIRONMENT ------------------------")
    logger.info(f"Using {env.device} device.")
    if env.ddp:
        logger.info(f"Using DDP with {env.ddp_world_size} devices.")
    else:
        logger.info("Not using DDP.")

    if env.use_tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("Using tf32.")
    else:
        logger.info("Not using tf32.")

    if env.use_autocast:
        logger.info("Using autocast.")
    else:
        logger.info("Not using autocast.")
    logger.info("------------------------ ENVIRONMENT ------------------------")

    return env
