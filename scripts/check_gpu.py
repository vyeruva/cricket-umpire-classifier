"""Run this before anything else. Must print True + the card name.

If it prints False or errors with "no kernel image is available for execution
on the device", torch resolved a build without sm_120 (Blackwell) kernels —
reinstall with:
    pip uninstall torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"""

import torch

available = torch.cuda.is_available()
print("CUDA available:", available)

if available:
    print("Device:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")
    print("Matmul on GPU OK:", x.shape)
else:
    print("CUDA is NOT available — fix this before training. See docstring above.")
