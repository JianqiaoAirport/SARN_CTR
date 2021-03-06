# -*- coding: utf-8 -*-
import sys
# python 2.7
import cPickle
# python 3
# import pickle

data_dir = sys.argv[1] + "/"

f_train = open(data_dir+"training_set", "r")
uid_dict = {}
mid_dict = {}
cat_dict = {}

iddd = 0
for line in f_train:
    arr = line.strip("\n").split("\t")
    clk = arr[0]
    uid = arr[1]
    mid = arr[2]
    cat = arr[3]
    mid_list = arr[4]
    cat_list = arr[5]

    if uid not in uid_dict:
        uid_dict[uid] = 0
    uid_dict[uid] += 1
    if mid not in mid_dict:
        mid_dict[mid] = 0
    mid_dict[mid] += 1
    if cat not in cat_dict:
        cat_dict[cat] = 0
    cat_dict[cat] += 1
    if len(mid_list) == 0:
        continue
    for m in mid_list.split(""):
        if m not in mid_dict:
            mid_dict[m] = 0
        mid_dict[m] += 1

    iddd+=1
    for c in cat_list.split(""):
        if c not in cat_dict:
            cat_dict[c] = 0
        cat_dict[c] += 1
f_train.close()


print("#uid", len(uid_dict))
print("#mid", len(mid_dict))
print("#cat", len(cat_dict))

# 2.7
sorted_uid_dict = sorted(uid_dict.iteritems(), key=lambda x:x[1], reverse=True)
sorted_mid_dict = sorted(mid_dict.iteritems(), key=lambda x:x[1], reverse=True)
sorted_cat_dict = sorted(cat_dict.iteritems(), key=lambda x:x[1], reverse=True)
# 3.5
# sorted_uid_dict = sorted(uid_dict.items(), key=lambda x:x[1], reverse=True)
# sorted_mid_dict = sorted(mid_dict.items(), key=lambda x:x[1], reverse=True)
# sorted_cat_dict = sorted(cat_dict.items(), key=lambda x:x[1], reverse=True)


uid_voc = {}
index = 0
for key, value in sorted_uid_dict:
    uid_voc[key] = index
    index += 1
mid_voc = {}
mid_voc["default_mid"] = 0
index = 1
for key, value in sorted_mid_dict:
    mid_voc[key] = index
    index += 1
cat_voc = {}
cat_voc["default_cat"] = 0
index = 1
for key, value in sorted_cat_dict:
    cat_voc[key] = index
    index += 1


# python 2.7
cPickle.dump(uid_voc, open(data_dir+"uid_voc.pkl", "w"))
cPickle.dump(mid_voc, open(data_dir+"mid_voc.pkl", "w"))
cPickle.dump(cat_voc, open(data_dir+"cat_voc.pkl", "w"))
# python 3
# pickle.dump(uid_voc, open(data_dir+"uid_voc.pkl", "wb"))
# pickle.dump(mid_voc, open(data_dir+"mid_voc.pkl", "wb"))
# pickle.dump(cat_voc, open(data_dir+"cat_voc.pkl", "wb"))