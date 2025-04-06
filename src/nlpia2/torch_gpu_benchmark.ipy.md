>>> import torch
>>> torch.device('cuda')
device(type='cuda')
>>> torch.device('cuda:0')
device(type='cuda', index=0)
>>> torch.device('cuda:1')
device(type='cuda', index=1)
>>> torch.device('cuda:2')
device(type='cuda', index=2)
>>> torch.device('cuda:3')
device(type='cuda', index=3)
>>> torch.device('cuda:16')
device(type='cuda', index=16)
>>> x = torch.tensor(list(range(3)))
>>> x.to?
>>> x.device?
>>> import torch
... import math
... 
... 
... dtype = torch.float
... device = torch.device("cpu")
... # device = torch.device("cuda:0") # Uncomment this to run on GPU
... 
... # Create random input and output data
... x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
... y = torch.sin(x)
...
>>> y
tensor([ 8.7423e-08, -3.1430e-03, -6.2863e-03,  ...,  6.2863e-03,
         3.1432e-03, -8.7423e-08])
>>> import torch
... import math
... 
... 
... dtype = torch.float
... # device = torch.device("cpu")
... device = torch.device("cuda:0") # Uncomment this to run on GPU
... 
... # Create random input and output data
... x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
... y = torch.sin(x)
...
>>> y
tensor([ 8.7423e-08, -3.1430e-03, -6.2863e-03,  ...,  6.2863e-03,
         3.1430e-03, -8.7423e-08], device='cuda:0')
>>> import torch
... import math
... 
... 
... dtype = torch.float
... # device = torch.device("cpu")
... device = torch.device("cuda:1") # Uncomment this to run on GPU
... 
... # Create random input and output data
... x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
... y = torch.sin(x)
...
>>> y
tensor([ 8.7423e-08, -3.1430e-03, -6.2863e-03,  ...,  6.2863e-03,
         3.1430e-03, -8.7423e-08], device='cuda:1')
>>> y.numpy()
>>> y.detach().numpy()
>>> y.cpu().numpy()
array([ 8.7422777e-08, -3.1429797e-03, -6.2862537e-03, ...,
        6.2862537e-03,  3.1429797e-03, -8.7422777e-08], dtype=float32)
>>> import torch
... 
... 
... def batched_dot_mul_sum(a, b):
...     '''Computes batched dot by multiplying and summing'''
...     return a.mul(b).sum(-1)
... 
... 
... def batched_dot_bmm(a, b):
...     '''Computes batched dot by reducing to bmm'''
...     a = a.reshape(-1, 1, a.shape[-1])
...     b = b.reshape(-1, b.shape[-1], 1)
...     return torch.bmm(a, b).flatten(-3)
... 
... 
... # Input for benchmarking
... x = torch.randn(10000, 64)
... 
... # Ensure that both functions compute the same output
... assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))
...
>>> import timeit
... 
... t0 = timeit.Timer(
...     stmt='batched_dot_mul_sum(x, x)',
...     setup='from __main__ import batched_dot_mul_sum',
...     globals={'x': x})
... 
... t1 = timeit.Timer(
...     stmt='batched_dot_bmm(x, x)',
...     setup='from __main__ import batched_dot_bmm',
...     globals={'x': x})
... 
... print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
... print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
...
>>> x = torch.randn(10000, 1024, device='cuda')
... 
... t0 = timeit.Timer(
...     stmt='batched_dot_mul_sum(x, x)',
...     setup='from __main__ import batched_dot_mul_sum',
...     globals={'x': x})
... 
... t1 = timeit.Timer(
...     stmt='batched_dot_bmm(x, x)',
...     setup='from __main__ import batched_dot_bmm',
...     globals={'x': x})
... 
... # Ran each twice to show difference before/after warmup
... print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
... print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
... print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
... print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
...
>>> x = torch.randn(10000, 1024, device='cuda:1')
... 
... t0 = timeit.Timer(
...     stmt='batched_dot_mul_sum(x, x)',
...     setup='from __main__ import batched_dot_mul_sum',
...     globals={'x': x})
... 
... t1 = timeit.Timer(
...     stmt='batched_dot_bmm(x, x)',
...     setup='from __main__ import batched_dot_bmm',
...     globals={'x': x})
... 
... # Ran each twice to show difference before/after warmup
... print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
... print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
... print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
... print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
...
>>> from itertools import product
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... sizes = [1, 64, 1024, 10000]
... for b, n in product(sizes, sizes):
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n))
...     for num_threads in [1, 4, 16, 32]:
...         results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='mul/sum',
...         ).blocked_autorange(min_run_time=1))
...         results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='bmm',
...         ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> from import torch.utils import benchmark
>>> from torch.utils import benchmark
>>> from itertools import product
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... sizes = [1, 64, 1024, 10000]
... for b, n in product(sizes, sizes):
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n))
...     for num_threads in [1, 4, 16, 32]:
...         results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='mul/sum',
...         ).blocked_autorange(min_run_time=1))
...         results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='bmm',
...         ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> num_threads = torch.get_num_threads()
... print(f'Benchmarking on {num_threads} threads')
... 
... t0 = benchmark.Timer(
...     stmt='batched_dot_mul_sum(x, x)',
...     setup='from __main__ import batched_dot_mul_sum',
...     globals={'x': x},
...     num_threads=num_threads,
...     label='Multithreaded batch dot',
...     sub_label='Implemented using mul and sum')
... 
... t1 = benchmark.Timer(
...     stmt='batched_dot_bmm(x, x)',
...     setup='from __main__ import batched_dot_bmm',
...     globals={'x': x},
...     num_threads=num_threads,
...     label='Multithreaded batch dot',
...     sub_label='Implemented using bmm')
... 
... print(t0.timeit(100))
... print(t1.timeit(100))
...
>>> from itertools import product
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... for device in (torch.device(dev) for dev in 'cuda:1 cuda:0 cpu'.split()):
...   sizes = [1, 64, 1024]
...   for b, n in product(sizes, sizes):
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n), device=device)
...     for num_threads in [1, 4, 16, 32]:
...       results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='mul/sum',
...       ).blocked_autorange(min_run_time=1))
...       results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='bmm',
...       ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> from itertools import product
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... for device in (torch.device(dev) for dev in 'cuda:1 cuda:0 cpu'.split()):
...   print(device)
...   sizes = [1, 64]
...   for b, n in product(sizes, sizes):
...     print(f"{b}x{n}")
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n), device=device)
...     for num_threads in [1, 4, 16, 32]:
...       results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='mul/sum',
...       ).blocked_autorange(min_run_time=1))
...       results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description='bmm',
...       ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> from itertools import product
... from tqdm import tqdm
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... for device in (torch.device(dev) for dev in 'cuda:1 cuda:0 cpu'.split()):
...   print(device)
...   sizes = [1, 64]
...   for b, n in tqdm(list(product(sizes, sizes))):
... #    print(f"{b}x{n}")
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n), device=device)
...     for num_threads in [1, 4, 16, 32]:
...       results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             device=device,
...             sub_label=sub_label,
...             description='mul/sum',
...       ).blocked_autorange(min_run_time=1))
...       results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             device=device,
...             sub_label=sub_label,
...             description='bmm',
...       ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> from itertools import product
... from tqdm import tqdm
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... for device in (torch.device(dev) for dev in 'cuda:1 cuda:0 cpu'.split()):
...   print(device)
...   sizes = [1, 64]
...   for b, n in tqdm(list(product(sizes, sizes))):
... #    print(f"{b}x{n}")
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n), device=device)
...     for num_threads in [1, 4, 16, 32]:
...       results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description=f'mul/sum {device}',
...       ).blocked_autorange(min_run_time=1))
...       results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             device=device,
...             sub_label=sub_label,
...             description=f'bmm {device}',
...       ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> from itertools import product
... from tqdm import tqdm
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... for device in (torch.device(dev) for dev in 'cuda:1 cuda:0 cpu'.split()):
...   print(device)
...   sizes = [1, 64]
...   for b, n in tqdm(list(product(sizes, sizes))):
... #    print(f"{b}x{n}")
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n), device=device)
...     for num_threads in [1, 4, 16, 32]:
...       results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description=f'mul/sum {device}',
...       ).blocked_autorange(min_run_time=1))
...       results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description=f'bmm {device}',
...       ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> from itertools import product
... from tqdm import tqdm
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... for device in (torch.device(dev) for dev in 'cuda:1 cuda:0 cpu'.split()):
...   print(device)
...   sizes = [1, 1024, 32768]
...   for b, n in tqdm(list(product(sizes, sizes))):
... #    print(f"{b}x{n}")
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n), device=device)
...     for num_threads in [1, 16, 64]:
...       results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description=f'mul/sum {device}',
...       ).blocked_autorange(min_run_time=1))
...       results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description=f'bmm {device}',
...       ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> from itertools import product
... from tqdm import tqdm
... 
... # Compare takes a list of measurements which we'll save in results.
... results = []
... 
... for device in (torch.device(dev) for dev in 'cuda:1 cuda:0 cpu'.split()):
...   print(device)
...   sizes = [1, 1024, 4096]
...   for b, n in tqdm(list(product(sizes, sizes))):
... #    print(f"{b}x{n}")
...     # label and sub_label are the rows
...     # description is the column
...     label = 'Batched dot'
...     sub_label = f'[{b}, {n}]'
...     x = torch.ones((b, n), device=device)
...     for num_threads in [1, 16, 64]:
...       results.append(benchmark.Timer(
...             stmt='batched_dot_mul_sum(x, x)',
...             setup='from __main__ import batched_dot_mul_sum',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description=f'mul/sum {device}',
...       ).blocked_autorange(min_run_time=1))
...       results.append(benchmark.Timer(
...             stmt='batched_dot_bmm(x, x)',
...             setup='from __main__ import batched_dot_bmm',
...             globals={'x': x},
...             num_threads=num_threads,
...             label=label,
...             sub_label=sub_label,
...             description=f'bmm {device}',
...       ).blocked_autorange(min_run_time=1))
... 
... compare = benchmark.Compare(results)
... compare.print()
...
>>> hist -o -p -f torch_gpu_benchmark.ipy.md
Mon Nov  7 15:09:21 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:03:00.0 Off |                  N/A |
|  0%   50C    P8    13W / 250W |     16MiB /  6144MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
| 23%   37C    P8     9W / 250W |      6MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      3030      G   /usr/lib/xorg/Xorg                 13MiB |
|    1   N/A  N/A      3030      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
03:00.0 VGA compatible controller: NVIDIA Corporation GM200 [GeForce GTX 980 Ti] (rev a1)
03:00.1 Audio device: NVIDIA Corporation GM200 High Definition Audio (rev a1)
04:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)
04:00.1 Audio device: NVIDIA Corporation GP102 HDMI Audio Controller (rev a1)
