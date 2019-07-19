import _pickle as cP

fw = open("index_MSD.tsv","w") # where to save the file?

train_list = cP.load(open('filtered_list_train.cP','rb'))
test_list = cP.load(open('filtered_list_test.cP','rb'))
id7d_to_path = cP.load(open('7D_id_to_path.pkl','rb'))
idmsd_to_id7d = cP.load(open('MSD_id_to_7D_id.pkl','rb'))
idmsd_to_tag = cP.load(open('msd_id_to_tag_vector.cP','rb'))

for idmsd in train_list:
    path = idmsd[2]+'/'+idmsd[3]+'/'+idmsd[4]+'/'+idmsd+'.mp3'
    fw.write("%s\t%s\n" % (idmsd,path))

for idmsd in test_list:
    path = idmsd[2]+'/'+idmsd[3]+'/'+idmsd[4]+'/'+idmsd+'.mp3'
    fw.write("%s\t%s\n" % (idmsd,path))
 
# for loading a file (train_list[0])
# file_dir = id7d_to_path[idmsd_to_id7d[train_list[0]]]

# for loading 50 tag vector of (train_list[0])
# tag_boolean = idmsd_to_tag[train_list[0]]
print('Done!')
