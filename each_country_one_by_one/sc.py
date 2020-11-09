from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom


def main():
    
    for i in range(7):
        print('countrycountry:',argv[1:][0])
        subprocess.call("python ./prediction.py "+str(i)+" "+argv[1:][0], shell=True)
    subprocess.call("python ./errors.py "+argv[1:][0], shell=True)

if __name__ == "__main__":
    
    
    main()
