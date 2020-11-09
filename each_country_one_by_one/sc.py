from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom


def main():
    print('countrycountry:', argv[1:][0])
    arg = "'" + argv[1:][0] + "'"
    print('arg: ', arg)
    for i in range(7):
        subprocess.call("python ./prediction.py " + str(i) + " " + arg, shell=True)
    subprocess.call("python ./errors.py " + arg, shell=True)


if __name__ == "__main__":
    main()
