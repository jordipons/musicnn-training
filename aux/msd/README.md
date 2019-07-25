# MSD_split
MSD split that Jongpil Lee used for his experiments, see: https://github.com/jongpillee/music_dataset_split

The split is based on the split version in https://github.com/keunwoochoi/MSD_split_for_tagging.
The difference is that Jongpil only left audios with more than 29.1 seconds.

# File descriptions
filtered_list_train.cP: train list (I used first 201680 IDs for training set and rest 11774 IDs for validation) <br>
filtered_list_test.cP: test list (total 28435 MSD IDs)<br>
7D_id_to_path.pkl: dictionary (Keys: 7digital ID, Values: file_path (i.e. "8/9/8947470.clip.mp3") )<br>
MSD_id_to_7D_id.pkl: dictionary (Keys: MSD ID, Values: 7digital ID)<br>
msd_id_to_tag_vector.cP: dictionary (Keys: MSD ID, Values: 50 tag vectors)<br>
50tagList.txt: 50 tag list <br>

# Usage example:
In python, by using these commands we can simply use given data.
<pre><code>
import _pickle as cP
train_list = cP.load(open('filtered_list_train.cP','rb'))
valid_list = train_list[201680:]
train_list = train_list[0:201680]
test_list = cP.load(open('filtered_list_test.cP','rb'))
id7d_to_path = cP.load(open('7D_id_to_path.pkl','rb'))
idmsd_to_id7d = cP.load(open('MSD_id_to_7D_id.pkl','rb'))
idmsd_to_tag = cP.load(open('msd_id_to_tag_vector.cP','rb'))
 
# for loading a file (train_list[0])
dir_path = "YOUR PATH"
file_dir = dir_path + id7d_to_path[idmsd_to_id7d[train_list[0]]]
 
# for loading 50 tag vector of (train_list[0])
tag_boolean = idmsd_to_tag[train_list[0]]
</code></pre>
