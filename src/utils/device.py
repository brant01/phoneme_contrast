from typing import Optional, Literal
import torch
import platform
import os
import logging

def get_best_device(
    device_str: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    logger: Optional[logging.Logger] = None
) -> torch.device:
    """
    Determine the best device for computation.

    Args:
        device_str: One of 'auto', 'cuda', 'mps', or 'cpu'.
        logger: Optional logger for recording device selection.

    Returns:
        torch.device
    """
    log_info = logger.info if logger else print
    log_debug = logger.debug if logger else print

    if device_str != "auto":
        log_info(f"Using explicitly requested device: {device_str}")
        return torch.device(device_str)

    # Auto-selection
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        log_info(f"Found {device_count} CUDA device(s):")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            log_debug(f"  GPU {i}: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB")

        preferred_idx = 0
        preferred_props = torch.cuda.get_device_properties(preferred_idx)
        log_info(f"Selected NVIDIA GPU: {preferred_props.name}")
        return torch.device(f"cuda:{preferred_idx}")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log_info("Using Apple Silicon GPU with MPS")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        mac_version = platform.mac_ver()[0]
        if mac_version:
            major_version = int(mac_version.split('.')[0])
            if major_version < 13:
                log_debug(f"macOS {mac_version} detected. MPS performs best on macOS 13+")

        return torch.device("mps")

    else:
        log_info("No GPU found, using CPU")
        cpu_count = os.cpu_count()
        if cpu_count:
            optimal_threads = max(1, cpu_count - 2)
            torch.set_num_threads(optimal_threads)
            log_debug(f"Set PyTorch to use {optimal_threads} of {cpu_count} CPU threads")

        return torch.device("cpu")
