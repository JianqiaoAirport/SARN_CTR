import os
import sys
import random

import tempfile
from subprocess import call


def main(file, temporary=False):
    tf_os, tpath = tempfile.mkstemp(dir='tmp/')
    # print("tf_os", tf_os)
    # print("tpath", tpath)
    tf = open(tpath, 'w')

    fd = open(file, "r")
    for l in fd:
        print >> tf, l.strip("\n")
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)
    if temporary:
        path, filename = os.path.split(os.path.realpath(file))
        # print("path", path)
        # print("filename", filename)
        fd = tempfile.TemporaryFile(prefix=filename + '.shuf', dir=path)
        # fd = tempfile.NamedTemporaryFile(prefix=filename + '.shuf', dir=path, delete=False)
    else:
        fd = open(file + '.shuf', 'w')

    # counter = 0
    for l in lines:
        s = l.strip("\n")
        # if counter <= 6:
        #     print(s)
        #     counter += 1
        print >> fd, s

    if temporary:
        fd.seek(0)
    else:
        fd.close()

    os.remove(tpath)

    return fd


if __name__ == '__main__':
    main(sys.argv[1])

