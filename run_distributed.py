import subprocess
import time
import importlib
def main():
   
#    for i in range(5):
#         time.sleep(1)
#         print('\rCount down : {} s'.format(5 - 1 - i), end='')
    p = importlib.import_module('util.data')
    print(p.__dict__)

if __name__ == "__main__":
    main()
