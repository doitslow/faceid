import os
import sys
import os.path as op
sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
import csv
import argparse
# ===============================================================================

def sub_files(in_dir):
    return [op.join(in_dir, f) for f in sorted(os.listdir(in_dir))
            if op.isfile(op.join(in_dir, f))]

def write_tprs(tprs, out_file):
    with open(out_file, 'a') as fw:
        writer = csv.writer(fw)
        writer.writerow(['Model', 'AUC', 'E-7', 'E-6', 'E-5', 'E-4', 'E-3', 'E-2', 'E-1'])
        for file in tprs:
            with open(file) as f:
                reader = csv.reader(f)
                count = 0
                for line in reader:
                    if count == 1:
                        writer.writerow(line)
                    count += 1
            f.close()
        writer.writerow([''])
        fw.close()

def summ_one_folder(in_dir):
    files = sub_files(in_dir)
    adience_tprs = [i for i in files if i.endswith('Adience_v0.0.1-tprs.csv')]
    ijbc_tprs = [i for i in files if i.endswith('IJBC_v0.0.2-tprs.csv')]
    morph_tprs = [i for i in files if i.endswith('MORPHacademic_v0.0.2-tprs.csv')]
    out_file = op.join(in_dir, '2-tprs_summ.csv')

    write_tprs(adience_tprs, out_file)
    write_tprs(ijbc_tprs, out_file)
    write_tprs(morph_tprs, out_file)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--in_dir', type=str, help='directory to the evaluation folder')
args = parser.parse_args()

if __name__ == '__main__':
    summ_one_folder(args.in_dir)