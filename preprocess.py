import numpy as np
from pycasia import CASIA
import lmdb
import io
import os

casia = CASIA.CASIA(path="/home/featurize/data/HWDB/")
train_iter = casia.load_dataset("train")

env = lmdb.open("/home/featurize/data/HWDB/lmdb", map_size=25951162777)

label2id = {}
id2label = {}
count = 0
label_count = {}
txn = env.begin(write=True)
total_count = 0
train_filelist = []
for (i, l) in train_iter:
    l = l.rstrip("\x00")
    if l not in label2id:
        label2id[l] = count
        id2label[count] = l
        count += 1
        label_count[l] = 0
    label_count[l] = label_count[l] + 1
    img_bytes = io.BytesIO()
    i.convert("RGB").save(img_bytes, format="JPEG")
    sample_name = "train/" + str(label2id[l]) + "-" + str(label_count[l])
    train_filelist.append({"name": sample_name, "label": l})
    txn.put(sample_name.encode(), img_bytes.getvalue())
    total_count += 1
    if total_count % 10000 == 0:
        txn.commit()
        txn = env.begin(write=True)

test_filelist = []
test_list = os.listdir("/home/featurize/data/HWDB/test")
for filename in test_list:
    try:
        test_iter = casia.load_gnt_file("/home/featurize/data/HWDB/test/" + filename)
        for (i, l) in test_iter:
            l = l.rstrip("\x00")
            label_count[l] = label_count[l] + 1
            img_bytes = io.BytesIO()
            i.convert("RGB").save(img_bytes, format="JPEG")
            sample_name = "test/" + str(label2id[l]) + "-" + str(label_count[l])
            test_filelist.append({"name": sample_name, "label": l})
            txn.put(sample_name.encode(), img_bytes.getvalue())
            total_count += 1
            if total_count % 10000 == 0:
                txn.commit()
                txn = env.begin(write=True)
    except:
        continue
txn.commit()


np.savez(
    "/home/featurize/data/HWDB/preprocess.npz",
    label2id=label2id,
    id2label=id2label,
    train_filelist=train_filelist,
    test_filelist=test_filelist,
)
