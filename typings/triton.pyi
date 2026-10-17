# Stub for `triton`, which torch installs only on linux (CUDA wheels).
#
# scripts/inference.py imports it inside the CUDA-only fla decode-kernel path;
# this covers the two helpers it calls, so pyright passes on a Mac too.

def next_power_of_2(n: int) -> int: ...
def cdiv(x: int, y: int) -> int: ...
