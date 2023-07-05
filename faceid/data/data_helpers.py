import os
import json


def load_data_info(data_root, finfo):
    full_datainfo = json.load(open(finfo))
    train_info, valid_info = full_datainfo["Train"], full_datainfo["Valid"]
    for i, d in enumerate(train_info):
        d['data_dir'] = os.path.join(data_root, d['data_dir'] )
        d['list_path'] = os.path.join(d['data_dir'], d['list_path'])
        d['is_train'] = True
    for i, d in enumerate(valid_info):
        d['data_dir'] = os.path.join(data_root, d['data_dir'])
        d['list_path'] = os.path.join(d['data_dir'], d['list_path'])
        d['is_train'] = False

    return train_info, valid_info