import random

line_pairs = []

with open("fens.yaml", "r") as source:
    for line in source:
        if line.startswith("- "):
            line_pairs[-1].append(line)
        else:
            line_pairs.append([line])


random.shuffle(line_pairs)

with open("fens_smol.yaml", "w") as destination:
    for pair in line_pairs[:20000]:
        destination.write(pair[0])
        destination.write(pair[1])
