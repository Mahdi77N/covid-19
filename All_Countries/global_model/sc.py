from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom

def Run(i):
    subprocess.call("python ./all_countries_one_model_prediction.py "+str(i), shell=True)

def main():
    loom = ProcessLoom(max_runner_cap = 8)
    for r in range(1,10):
        
        print('r = ',r)
        Run(r)
#         loom.add_function(Run,[i])
#     loom.execute()

        
if __name__ == "__main__":
    
    main()