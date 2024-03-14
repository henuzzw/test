import subprocess
from tqdm import tqdm
import time
import os, sys
import pickle
project = sys.argv[1]
sys.argv.append(project)
card = [0]
lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
#singlenums = {'Time':5, 'Math':2, "Lang":10, "Chart":3, "Mockito":4, "Closure":1, 'grace2811':1,'filtered_file':100}
singlenum = int(len(lst)/10) + 1  #singlenums[project]
print(sys.version)
print(sys.version_info)
print(sys.executable)
subprocess.Popen("python --version", shell=True).wait()
cards=len(card)
totalnum = len(card)*singlenum
lr = 1e-2
seed = 0
batch_size = 60
modelName="GcnNet"
layers=6
for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in card:
        cardn =j

        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + "  /home/wushumei/.conda/envs/pytorch/bin/python ./run.py %d %s %f %d %d %s %d"%(i, project, lr, seed, batch_size,modelName,layers), shell=True)
        jobs.append(p)
        time.sleep(10)
    for p in jobs:
        p.wait()
p = subprocess.Popen("/home/wushumei/.conda/envs/pytorch/bin/python sum.py %s %d %f %d %s"%(project, seed, lr, batch_size,str(layers)+modelName), shell=True)
p.wait()


