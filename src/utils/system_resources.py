# src/utils/system_resources.py

import torch
import multiprocessing
from typing import Optional
import multiprocessing
from torch import cuda
import logging

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

def adjust_exp_params_for_system(params: ExpParams, 
                                 device: torch.device, 
                                 logger: Optional[logging.Logger] = None) -> ExpParams:


    def log(msg): 
        logger.debug(msg) if logger else print(msg)

    log("Adjusting experiment parameters based on system resources...")

    # Threading
    num_cores = multiprocessing.cpu_count()
    params.num_workers = min(4, num_cores - 1)
    log(f"Detected CPU cores: {num_cores} -> num_workers set to {params.num_workers}")

    # GPU memory (if applicable)
    if device.type == "cuda":
        mem_total = cuda.get_device_properties(device).total_memory / (1024**3)
        log(f"GPU memory (GB): {mem_total:.2f}")

    # Pin memory
    params.pin_memory = device.type == "cuda"
    log(f"Pin memory: {params.pin_memory}")

    return params


def suggest_cluster_resources(params: ExpParams, model_size_gb: float = 1.5) -> dict:
    """
    Suggest reasonable cluster resource requests based on batch size and model size.
    """
    audio_ram_estimate = params.batch_size * 0.1  # assume 100MB per batch item max
    buffer = 2.0  # safety buffer in GB
    total_ram = audio_ram_estimate + model_size_gb + buffer

    gpu_mem = params.batch_size * 0.25 + model_size_gb  # conservative

    return {
        "cpus": min(16, multiprocessing.cpu_count()),
        "mem_gb": int(total_ram + 1),
        "gpus": 1 if params.device in ["cuda", "mps"] else 0,
        "gpu_mem_gb": int(gpu_mem + 1)
    }
