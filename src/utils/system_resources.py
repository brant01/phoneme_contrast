# src/utils/system_resources.py
import torch
import multiprocessing
from typing import Optional
import logging
from omegaconf import DictConfig, OmegaConf


def get_optimal_num_workers(safety_margin: int = 2) -> int:
    """
    Determine the optimal number of DataLoader workers based on CPU count.
    Leaves a few cores free for the OS and other processes.
    """
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count - safety_margin)


def get_optimal_batch_size(device: torch.device, base_size: int = 4) -> int:
    """
    Return a conservative-to-aggressive batch size estimate based on the device.
    """
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(device).total_memory
        if total_mem > 16 * 1024**3:
            return base_size * 8  # high memory GPU
        elif total_mem > 8 * 1024**3:
            return base_size * 4
        else:
            return base_size * 2
    elif device.type == "mps":
        return base_size * 2  # Apple Silicon is efficient
    else:
        return base_size  # CPU fallback


def adjust_params_for_system(
    cfg: DictConfig,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> DictConfig:
    """
    Adjust configuration based on system resources.
    
    Args:
        cfg: Hydra configuration
        device: PyTorch device
        logger: Logger instance
        
    Returns:
        Updated configuration
    """
    def log(msg):
        if logger:
            logger.debug(msg)
        else:
            print(msg)
    
    log("Adjusting parameters based on system resources...")
    
    # Make config writable
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    
    # Adjust number of workers
    num_cores = multiprocessing.cpu_count()
    suggested_workers = min(4, num_cores - 1)
    if cfg.training.num_workers == 0:
        cfg.training.num_workers = suggested_workers
    log(f"CPU cores: {num_cores}, num_workers: {cfg.training.num_workers}")
    
    # Adjust pin_memory based on device
    if cfg.training.pin_memory is None:
        cfg.training.pin_memory = (device.type == "cuda")
    log(f"Pin memory: {cfg.training.pin_memory}")
    
    # Adjust batch size based on GPU memory (if needed)
    if device.type == "cuda" and cfg.training.get('auto_batch_size', False):
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        if gpu_mem < 8:
            cfg.training.batch_size = min(cfg.training.batch_size, 16)
            log(f"Limited GPU memory ({gpu_mem:.1f}GB), batch_size: {cfg.training.batch_size}")
    
    return cfg


def suggest_cluster_resources(cfg: DictConfig, model_size_gb: float = 1.5) -> dict:
    """
    Suggest reasonable cluster resource requests based on config.
    """
    batch_size = cfg.training.get('batch_size', 16)
    device = cfg.get('device', 'cuda')
    
    audio_ram_estimate = batch_size * 0.1  # assume 100MB per batch item max
    buffer = 2.0  # safety buffer in GB
    total_ram = audio_ram_estimate + model_size_gb + buffer

    gpu_mem = batch_size * 0.25 + model_size_gb  # conservative

    return {
        "cpus": min(16, multiprocessing.cpu_count()),
        "mem_gb": int(total_ram + 1),
        "gpus": 1 if device in ["cuda", "mps"] else 0,
        "gpu_mem_gb": int(gpu_mem + 1)
    }