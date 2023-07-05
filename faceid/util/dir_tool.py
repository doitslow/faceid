import os
import shutil
from os.path import isdir, isfile, dirname, basename, join, exists

def mk_dir(dir_):
    if not exists(dir_):
        os.mkdir(dir_)

def rm_dir(dir_):
    if exists(dir_):
        os.remove(dir_)

def sub_dpaths(dir_):
    return [join(dir_, d) for d in sorted(os.listdir(dir_)) if isdir(join(dir_, d))]

def sub_dnames(dir_):
    return [d for d in sorted(os.listdir(dir_)) if isdir(join(dir_, d))]

def sub_fpaths(dir_):
    return [join(dir_, f) for f in sorted(os.listdir(dir_)) if isfile(join(dir_, f))]

def sub_fnames(dir_):
    return [f for f in sorted(os.listdir(dir_)) if isfile(join(dir_, f))]

def lowest_dpaths(dir_):
    _dirs = []
    for root, dirs, files in os.walk(dir_):
        if not dirs:
            _dirs.append(root)
    return _dirs

def all_fnames(dir_):
    fns = []
    for r, d, f in os.walk(dir_):
        for file in f:
            fns.append(file)
    return fns

def all_fpaths(dir_):
    files = []
    for r, d, f in os.walk(dir_):
        for file in f:
            files.append(os.path.join(r, file))
    return files
