try:
    import torch
    from . import _C
except ImportError:
    print(f"Failed to import fast_kernel._C")
    raise
