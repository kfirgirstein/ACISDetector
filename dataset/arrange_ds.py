import os
import sys
import json

if len(sys.argv) != 2:
    PATH = "./"
else:
    PATH = sys.argv[1] 

print(f"Start creating dictionary for json\nDirectory is {PATH}")

final_dataset = dict()
for d in  os.listdir(PATH):
    if os.path.isdir(f"{PATH}\\{d}"):
        samples = []
        for entry in os.listdir(f"{PATH}\\{d}"):
            with open(f"{PATH}/{d}/{entry}","rb") as f:
                buff = f.read()
            samples.append(list(buff))
        final_dataset[d] = samples
        print(f"Architecture {d} added successfully!")

print(f"Dictionary is ready to dump\nSize {len(final_dataset)}")

with open("./binary_raw.json","w") as fj:
    json.dump(final_dataset,fj)

print("Bye Bye!")