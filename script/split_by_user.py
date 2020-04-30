# -*- coding: utf-8 -*-

import sys
import random

save_dir = sys.argv[1] + "/"

fi = open(save_dir+"local_records", "r")
ftrain = open(save_dir+"training_set", "w")
ftest = open(save_dir+"test_set", "w")

while True:
    rand_int = random.randint(1, 10)
    noclk_line = fi.readline().strip()
    clk_line = fi.readline().strip()
    if noclk_line == "" or clk_line == "":
        break
    if rand_int == 1:
        ftest.write(noclk_line + '\n')
        ftest.write(clk_line + '\n')
    else:
        ftrain.write(noclk_line + '\n')
        ftrain.write(clk_line + '\n')
ftest.close()
ftrain.close()
