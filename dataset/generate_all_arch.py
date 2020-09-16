import os
import sys
if len(sys.argv) != 2:
    PATH = "./"
else:
    PATH = sys.argv[1] ##check if empty then current

for d in  os.listdir(PATH):
    if os.path.isdir(d):
        os.system(f'python generate_DS.py ./{d}')


