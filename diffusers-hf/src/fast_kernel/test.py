import os
import glob

extensions_dir = os.path.join("fast_kernel", "csrc")
print(f"extensions_dir: {extensions_dir}")
print(f"extensions_dir: {os.path.join(extensions_dir, '**', '*.cpp')}")
sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
print(f"sources: {sources}")

cuda_sources = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
print(f"cuda_sources: {cuda_sources}")

sources += cuda_sources

print(f"sourcse: {sources}")
