import sys
import argparse
from os.path import join, dirname, abspath
# avoid using relative path like '..' which does not seem to work with python3
parent_folder = dirname(dirname(abspath(__file__)))
sys.path.append(parent_folder)
# from util import dir_tool
from util import dir_tool


#=========#=========#=========#=========#=========#=========#=========#=========
def parse_args():
    parser = argparse.ArgumentParser("Delete unwanted checkpoints and evaluation files")
    parser.add_argument('--keep', type=str, nargs='+',
                        help='The sequence number of the items to be deleted')
    parser.add_argument('--din', '-d', type=str,
                        help='The directory to act on')
    return parser.parse_args()

def rm_unwanted(_dir, _seq):
    file_names = dir_tool.sub_fnames(_dir)
    ckpt_names = [f for f in file_names if f.endswith('.pth.tar')]
    eval_names = [f for f in file_names
                  if f.endswith('-roc.csv') or f.endswith('-tprs.csv')]
    for i in ckpt_names:
        if i.replace('.pth.tar', '').split('-')[-1] not in _seq:
            dir_tool.rm_dir(join(_dir, i))
    for i in eval_names:
        if i.split('-')[-3] not in _seq:
            dir_tool.rm_dir(join(_dir, i))

args = parse_args()
rm_unwanted(args.din, args.keep)