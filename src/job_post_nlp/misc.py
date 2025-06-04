import multiprocessing
import os

import psutil

multiprocessing.cpu_count()

os.cpu_count()


psutil.virtual_memory()


'''
with
batch_size: 50
threads: 4
it takes around 200-300it/s on star
'''
