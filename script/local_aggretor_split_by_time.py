# -*- coding: utf-8 -*-
import sys

data_dir = sys.argv[1]

fin = open(data_dir + "jointed-new-split-info-test", "r")
ftrain = open(data_dir + "local_train_splitByTime-test", "w")
ftest = open(data_dir + "local_test_splitByTime-test", "w")

last_user = "0"
common_fea = ""
line_idx = 0

for line in fin:
    items = line.strip().split("\t")
    ds = items[0]
    clk = int(items[1])
    user = items[2]
    movie_id = items[3]
    dt = items[-2]
    cat1 = items[-1]

    if user != last_user:
        movie_id_list = []
        cate1_list = []
        tim_list = []

    else:
        history_clk_num = len(movie_id_list)
        cat_str = ""
        mid_str = ""

        for c1 in cate1_list:
            cat_str += c1 + ""
        for mid in movie_id_list:
            mid_str += mid + ""

        if len(cat_str) > 0: cat_str = cat_str[:-1]
        if len(mid_str) > 0: mid_str = mid_str[:-1]

        # 对于electronics， 1388620800
        # 对于books，1389225600
        if int(dt) > 1388620800:  # 8 is the average length of user behavior
            if history_clk_num >= 4:
                ftest.write(items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 + "\t"  + mid_str + "\t" + cat_str + "\n")
        else:
            if history_clk_num >= 4:  # 8 is the average length of user behavior
                ftrain.write(items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 + "\t"  + mid_str + "\t" + cat_str + "\n")
    last_user = user
    if clk:
        movie_id_list.append(movie_id)
        cate1_list.append(cat1)
    line_idx += 1

ftest.close()
ftrain.close()
fin.close()
