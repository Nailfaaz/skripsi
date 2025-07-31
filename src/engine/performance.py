# src/engine/performance.py
import os
import time
import tempfile
import torch
import numpy as np
from ptflops import get_model_complexity_info

def measure_performance_metrics(model, loader, device, n_samples=100):
    model.eval()
    times, memory_usage = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            if len(times) >= n_samples: break
            imgs = imgs.to(device)
            if not times:
                [model(imgs) for _ in range(10)]
                if device.type == 'cuda': torch.cuda.synchronize()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated()
            start = time.time()
            model(imgs)
            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.time()
            if device.type == 'cuda':
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append((mem_after - mem_before) / (1024**2))
            batch_time = (end - start) / imgs.size(0)
            times.extend([batch_time] * imgs.size(0))
    avg_time_ms = np.mean(times[:n_samples]) * 1000
    throughput = 1000 / avg_time_ms
    avg_memory_mb = np.mean(memory_usage) if memory_usage else 0
    return {
        'inference_time_ms': avg_time_ms,
        'throughput_samples_per_sec': throughput,
        'memory_per_sample_mb': avg_memory_mb
    }

def get_comprehensive_model_stats(model, input_size=(1,224,224)):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)
    size_mb = os.path.getsize(tmp.name) / (1024*1024)
    tmp.close(); os.remove(tmp.name)
    flops, params_str = get_model_complexity_info(model, input_size, as_strings=(True, False), 
                                                  print_per_layer_stat=False, verbose=False)
    layer_count = len(list(model.modules()))
    conv_layers = len([m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))])
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': size_mb,
        'flops': flops[0] if isinstance(flops, tuple) else flops,
        'flops_human': flops[1] if isinstance(flops, tuple) else params_str,
        'total_layers': layer_count,
        'conv_layers': conv_layers,
        'parameters_human': params_str
    }
