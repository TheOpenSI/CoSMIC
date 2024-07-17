import fileinput

with fileinput.FileInput("raft.txt", inplace=True, backup='.bak') as file:
    for line in file:
        if "assistant: " in line:
            print(line.replace("assistant: ", ""), end="")
        else:
            print(line, end = "")