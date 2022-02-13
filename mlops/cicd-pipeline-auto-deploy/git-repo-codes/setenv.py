import yaml
import sys

args = sys.argv
num = len(args)

with open('flow.yml') as file:
    obj = yaml.safe_load(file)
    val = obj
    idx = 1
    while(idx < num):
        val = val[args[idx]]
        idx += 1
    print(val)
