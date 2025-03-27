import torch
import subprocess


def get_cuda_info():
    print(f'PyTorch version: {torch.__version__}')
    print('*'*10)
    print('_CUDA version: ')
    try:
        cuda_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        print(f'CUDA version:\n{cuda_version}')
    except FileNotFoundError:
        print('CUDA is not installed or nvcc is not in PATH')
    print('*'*10)
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name()}')
