import os
from os.path import basename, dirname, join, exists

din = '/raid/shared/face_data_202010/emore/20190710_Emore'
fout = '/raid/shared/face_data_202010/emore/emore_all.lst'

# with open(fout, 'w') as fopen:
#     for root, dirs, files in os.walk(din):
#         for file in files:
#             print(root)
#             lbl = str(int(basename(root).replace('.0', '')))
#             img_path = '20190710_Emore' + join(root.replace(din, ''), file)
#             fopen.write(lbl + '\t' + img_path +'\n')

with open(fout, 'r') as fopen:
    lines = fopen.readlines()
    for line in lines:
        img_path = line.strip('\n').split('\t')[-1]
        if not exists(join('/raid/shared/face_data_202010/emore', img_path)):
            print(img_path)

