"""Device selection and transfer helpers."""

from __future__ import annotations

import torch


class DeviceManager:
    """Manage CPU, CUDA, and MPS device selection."""

    def __init__(self, device_id=None):
        self.device_id = device_id
        if self.device_id == -1:
            self.device = torch.device("cpu")
        else:
            self.device = self._get_device()
        self._setup_device()

    def _get_device(self):
        if torch.cuda.is_available():
            return self._get_cuda_device()
        if torch.backends.mps.is_available():
            print("使用 MPS 设备")
            return torch.device("mps")
        print("警告: CUDA/MPS 不可用，使用 CPU")
        return torch.device("cpu")

    def _get_cuda_device(self):
        if self.device_id is not None:
            if self.device_id < torch.cuda.device_count():
                print(f"使用指定GPU: {self.device_id} ({torch.cuda.get_device_name(self.device_id)})")
                return torch.device(f"cuda:{self.device_id}")
            print(f"警告: GPU {self.device_id} 不存在，使用GPU 0")
            return torch.device("cuda:0")
        return self._select_best_gpu()

    def _select_best_gpu(self):
        best_gpu = 0
        max_free_memory = 0

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)
            free_memory = total_memory - allocated - cached

            print(f"  GPU {i}: {props.name}")
            print(f"    总内存: {total_memory / 1e9:.2f} GB")
            print(f"    已分配: {allocated / 1e9:.2f} GB")
            print(f"    缓存: {cached / 1e9:.2f} GB")
            print(f"    可用: {free_memory / 1e9:.2f} GB")

            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i

        print(f"\n自动选择GPU {best_gpu}，可用内存: {max_free_memory / 1e9:.2f} GB")
        return torch.device(f"cuda:{best_gpu}")

    def _setup_device(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(42)

    def get_device(self):
        return self.device

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        if isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        if hasattr(data, "to"):
            return data.to(self.device, non_blocking=True)
        return data

    def memory_info(self):
        if self.device.type == "cuda":
            device_id = self.device.index if self.device.index is not None else 0
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            max_allocated = torch.cuda.max_memory_allocated(device_id)
            return {
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "max_allocated_gb": max_allocated / 1e9,
                "device": torch.cuda.get_device_name(device_id),
            }
        return {"device": "cpu"}

if __name__ == "__main__":
    device_manager = DeviceManager()
    print(f"当前设备: {device_manager.get_device()}")
    print("内存信息:", device_manager.memory_info())
    