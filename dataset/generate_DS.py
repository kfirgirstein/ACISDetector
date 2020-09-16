import os
import sys
import random

# Yield successive n-sized 
# chunks from l. 
def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]      

TOTAL_SIZE = 5000
BATCH_SIZE = 1000
if len(sys.argv) != 2:
    PATH = "./"
else:
    PATH = sys.argv[1] ##check if empty then current

print(f"Start to create ds for PATH {PATH}")
dir_list =  os.listdir(PATH)
samples = []
while len(samples)<TOTAL_SIZE or len(dir_list)==0 :
    #print(len(samples))
    filename = random.choice(dir_list)
    dir_list.remove(filename)
    filename = f"{PATH}/{filename}"
    if filename.endswith(".code"):
        with open(filename,"rb") as f:
            buff = f.read()
            divided_buff = list(divide_chunks(buff,BATCH_SIZE))
            if len(divided_buff[-1])<1000:
                divided_buff.remove(divided_buff[-1])
            samples.extend(divided_buff)

print(f"All Sampled generated.. start saving them into files!")
samples = random.sample(samples,TOTAL_SIZE)
new_path = PATH+"/Dataset"

if not os.path.isdir(new_path):
    os.makedirs(new_path) 
    for i in range(len(samples)):
        with open(f"{new_path}/{i}","wb") as ff:
            ff.write(samples[i])
print(f"Bye Bye!")
