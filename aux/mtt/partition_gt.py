IDS_ERROR = set()
# NO FILE   /6/norine_braun-now_and_zen-08-gently-117-146.mp3
# CORRUPTED /8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3
# CORRUPTED /9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3
IDS_ERROR.add('35644')
IDS_ERROR.add('55753')
IDS_ERROR.add('57881')                        


def load_id2any(index_file, format=None):
    fspec = open(index_file)
    ids = []
    id2any = dict()
    for line in fspec.readlines():
        id, any = line.strip().split("\t")
        ids.append(id)
        if format == 'toFloat':
            id2any[id] = [float(i) for i in eval(any)]
        else:
            id2any[id] = any
    return ids, id2any
    

def split_magna(ids, id2path):
    train_set = []
    val_set = []
    test_set = []
    for id in ids:
        path = id2path[id]
        folder = int(path[path.rfind("/") - 1:path.rfind("/")], 16)
        if folder < 12:
            train_set.append(id)  # 0,1,2,3,4,5,6,7,8,9,a,b
        elif folder < 13:
            val_set.append(id)  # c
        else:
            test_set.append(id)  # d,e,f
    return train_set, val_set, test_set
    
    
def write_gt_file(ids, id2gt, file_name):
    fw = open(file_name,"w")
    for id in ids:
        if id in IDS_ERROR:
            continue
        fw.write("%s\t%s\n" % (id,id2gt[id]))
    fw.close()
        
    
ids, id2path = load_id2any('index_MAGNA.tsv')
train_ids, val_ids, test_ids = split_magna(ids, id2path)
_, id2gt = load_id2any('gt_classes_MAGNA50.tsv', format='toFloat')

print('# Train:',len(train_ids))
print('# Val:',len(val_ids))
print('# Train:',len(test_ids))

write_gt_file(train_ids, id2gt, 'train_gt_mtt.tsv')
write_gt_file(val_ids, id2gt, 'val_gt_mtt.tsv')
write_gt_file(test_ids, id2gt, 'test_gt_mtt.tsv')
