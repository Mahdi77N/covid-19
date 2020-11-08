from sys import argv
import sys
import os
import subprocess

def Run(i):
    # print('python ./{}/sc.py'.format(i))
    subprocess.call('python ./{}/sc.py'.format(i), shell=True)

def main():
    for i in range(1, 11):
        print(200 * '#')
        print(i)
        Run(i)


if __name__ == "__main__":

    main()
