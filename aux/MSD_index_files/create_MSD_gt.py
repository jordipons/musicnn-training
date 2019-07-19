import _pickle as cP
import numpy as np
import json

fw_train = open("train_gt_MSD.tsv","w") # where to save the file?
fw_val = open("val_gt_MSD.tsv","w") # where to save the file?
fw_test = open("test_gt_MSD.tsv","w") # where to save the file?

train_list = cP.load(open('filtered_list_train.cP','rb'))
val_list = train_list[201680:]
train_list = train_list[0:201680]
test_list = cP.load(open('filtered_list_test.cP','rb'))
id7d_to_path = cP.load(open('7D_id_to_path.pkl','rb'))
idmsd_to_id7d = cP.load(open('MSD_id_to_7D_id.pkl','rb'))
idmsd_to_tag = cP.load(open('msd_id_to_tag_vector.cP','rb'))

for idmsd in train_list:
    gt = np.squeeze(idmsd_to_tag[idmsd]).astype(int) 
    fw_train.write(str(idmsd) + '\t' + str(gt.tolist()) + '\n')

print('Train, done!')

for idmsd in val_list:
    gt = np.squeeze(idmsd_to_tag[idmsd]).astype(int) 
    fw_val.write(str(idmsd) + '\t' + str(gt.tolist()) + '\n')

print('Validation, done!')

for idmsd in test_list:
    gt = np.squeeze(idmsd_to_tag[idmsd]).astype(int) 
    fw_test.write(str(idmsd) + '\t' + str(gt.tolist()) + '\n')

print('ALL done!')




