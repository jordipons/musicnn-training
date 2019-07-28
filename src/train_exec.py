import os
import numpy as np
import time
import subprocess
import sys

setups = ['spec', 'spec', 'spec']
GPU = 0
script = 'train.py'

if __name__ == '__main__':
    start = time.time()
    for stp in setups:
        str_exec = 'CUDA_VISIBLE_DEVICES=' + str(GPU) + ' python ' + str(script) + ' ' + str(stp)
        #str_exec = 'CUDA_VISIBLE_DEVICES=' + str(GPU) + ' python3 ' + str(script) + ' ' + str(stp)
        print(str_exec)
        
        try:
            retcode = subprocess.call(str_exec, shell=True)
            if retcode < 0:
                print("Terminated by signal", retcode)
            else:
                print("Returned", retcode)
        except OSError as e:
                print("Execution failed:", e)

    end = time.time()
    print('\nDone! It took: %7.2f hours' % ((end - start) / 3600.0))
