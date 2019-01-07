import sys

bp1 = set()
bp2 = set()

with open(sys.argv[1]) as f:
    bp1 = set([l for l in f])

with open(sys.argv[2]) as f:
    bp2 = set([l for l in f])

common = bp1.intersection(bp2)

for p in common:
    print(p.strip())
