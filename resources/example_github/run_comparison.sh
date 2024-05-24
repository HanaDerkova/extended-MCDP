#!/bin/bash

python ../../src/mcdp.py tcga_intervals.txt hirt_intervals.txt chr_sizes.txt -m hankas_method --log hirt.log --sf pvalues_hirt.txt -i trained_model_intervals_hirt.txt -g trained_model_hirt_gaps.txt
