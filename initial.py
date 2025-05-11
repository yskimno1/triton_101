import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(X, Y, Z, N):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(X + idx, mask=mask)
    y = tl.load(Y + idx, mask=mask)
    z = x + y
    tl.store(Z + idx, z, mask=mask)

BLOCK_SIZE = 1024
N = 4096
x = torch.rand(N, device='cpu')
y = torch.rand(N, device='cpu')
z = torch.empty_like(x)

add_kernel[(N // BLOCK_SIZE, )](x, y, z, N)
