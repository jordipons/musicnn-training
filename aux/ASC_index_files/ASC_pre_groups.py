import pandas as pd
import numpy as np

metadata_train_path = "meta_dev.tsv" # where is the original metadata file?
metadata_test_path = "meta_eval.tsv" # where is the original metadata file?
name = "asc_fold" # name of the output file?
fold = 0

#classes = ['beach', 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram']

# Groups:
group_0 = ['bus', 'car']
group_1 = ['office', 'library', 'home', 'cafe/restaurant']
group_2 = ['tram', 'train']
group_3 = ['forest_path', 'residential_area', 'park', 'city_center']
## OUT: beach, metro_station, grocery_store

num_classes = 4 # number of groups

def get_id_group(class_name):
    if class_name in group_0:
        return True, 0
    elif class_name in group_1:
        return True, 1
    elif class_name in group_2:
        return True, 2
    elif class_name in group_3:
        return True, 3
    else:
        return False, False

train_items = pd.read_csv(metadata_train_path, delimiter='\t', header=None)
test_items = pd.read_csv(metadata_test_path, delimiter='\t', header=None)

index_w = open("index_"+name+"s_all.tsv", "w")
gt_tr_w = open("gt_"+name+str(fold)+"_train.tsv", "w")
gt_test_w = open("gt_"+name+str(fold)+"_test.tsv", "w")

for index_train, row in train_items.iterrows():
    cnd, idx = get_id_group(row[1])
    if cnd:
        gt_vector = np.zeros(num_classes)
        gt_vector[idx] = 1
        gt_tr_w.write("%s\t%s\n" % (index_train,gt_vector.tolist()))
        index_w.write("%s\t%s\n" % (index_train,"TUT-acoustic-scenes-2017-development/"+row[0]))

print(row)
print(index_train)

for index_test, row in test_items.iterrows():
    cnd, idx = get_id_group(row[1])
    if cnd:
        index = index_train + 1 + index_test
        index_w.write("%s\t%s\n" % (index,"TUT-acoustic-scenes-2017-evaluation/"+row[0]))
        gt_vector = np.zeros(num_classes)
        gt_vector[idx] = 1
        gt_test_w.write("%s\t%s\n" % (index,gt_vector.tolist()))

print(index_test)
print(index)

print('Done!')
