#!/usr/bin/env python
import optparse
import sys

optparser = optparse.OptionParser()
optparser.add_option("-e", dest="e_file", default="data/hansards.e", help="E filename")
optparser.add_option("-f", dest="f_file", default="data/hansards.f", help="F filename")
(opts, args) = optparser.parse_args()

f_data = open(opts.f_file)
e_data = open(opts.e_file)

for (n, (f, e, a)) in enumerate(zip(f_data, e_data, sys.stdin)):
    size_f = len(f.strip().split())
    size_e = len(e.strip().split())
    try:
        alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
        for (i, j) in alignment:
            if (i >= size_f or j > size_e):
                sys.stderr.write(
                    "WARNING (%s): Sentence %d, point (%d,%d) is not a valid link\n" % (sys.argv[0], n, i, j))
                sys.exit(1)
            pass
    except (Exception):
        sys.stderr.write("ERROR (%s) line %d is not formatted correctly:\n  %s" % (sys.argv[0], n, a))
        sys.stderr.write(
            "Lines can contain only tokens \"i-j\", where i and j are integer indexes into the French and English sentences, respectively.\n")
        sys.exit(1)

warned = False
for a in (sys.stdin):
    if not warned:
        sys.stderr.write("WARNING (%s): alignment file is longer than bitext\n" % sys.argv[0])
        warned = True

try:
    if (next(f_data)):
        sys.stderr.write("WARNING (%s): bitext is longer than alignment\n" % sys.argv[0])
        warned = True
except (StopIteration):
    pass

if not warned:
    print("Data validation passed.")
