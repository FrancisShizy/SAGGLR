#!/bin/bash

LIST_TARGETS=`cat data/list_targets_SRC.txt`;

for target in ${LIST_TARGETS}; do
    echo "Now training on target w. PDB Id: ${target}"
    python -m SAGGLR.cross_valid --target ${target}
done
