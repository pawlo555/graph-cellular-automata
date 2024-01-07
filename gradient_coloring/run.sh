#!/bin/bash

export PYTHONPATH=$PYTHONPATH:"../`pwd`"

touch results.txt

for n in 10 20 30 40 50 60; do
  for e in 20 40 60 80 100; do
    echo "nodes: $n, edges: $e"
    python3 test.py --nodes $n --edges $e >> results.txt
  done
done

