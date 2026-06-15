import time
import os
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, required=True)
parser.add_argument("--global_seed", type=int, default=42)
args = parser.parse_args()


absolute_path = os.getcwd()

j = 0
while True:
    fin = open(os.path.join(absolute_path, args.command), "r")
    i = 0
    exit_or_not = False
    while True:
        try:
            if i < j:
                _ = fin.readline()
                i += 1
            else:
                content = fin.readline()
                if content[-1] == "\n":
                    content = content[:-1]
                if content == "exit":
                    exit_or_not = True
                    break
                elif content =="":
                    break
                else:
                    j += 1
                    break
        except:
            break
    if exit_or_not == True:
        break
    try:
        if content[:2] == "cd":
            if content[-1] == "\n":
                content = content[:-1]
            os.chdir(content[3:])
        elif content != "" and content != "exit":
            if content[-1] == "\n":
                content = content[:-1]
            # content += " &"
            content = "GLOBAL_SEED={} ".format(args.global_seed) + content
            os.system(content)
    except:
        print("Error!")
    time.sleep(1)
    fin.close()