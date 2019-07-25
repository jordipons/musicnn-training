import pandas as pd
import json
import csv
import numpy as np

use_code='jordi_50' #'sergio','jordi_180','jordi_50'

#doesn't matter if the first column has the tags info, because when doing id2gt we never load the id with a tag! is always an ID.

if use_code=='sergio':
    df_items = pd.read_table("/home/idrojsnop/ICML/annotations_final.csv")
    output_file_name = '/home/idrojsnop/ICML/bla.tsv'

    fw = open(output_file_name,"w")
    for index, row in df_items.iterrows():
        classes = json.dumps(row.ix[1:-1].tolist())
        id = str(row['clip_id'])
        fw.write("%s\t%s\n" % (id,classes))

elif use_code=='jordi_180':
    output_file_name = '/home/idrojsnop/ICML/jordi_180.tsv'
    original_file = "/home/idrojsnop/ICML/annotations_final.csv"

    fw = open(output_file_name,"w")
    with open(original_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        idx=-1
        for row in csv_reader:
            idx=idx+1
            print(idx)
            if idx!=0:
                classes = json.dumps(row[1:-1])
                id = str(row[0])
                fw.write("%s\t%s\n" % (id,classes))
            else:
                header=row[1:-1]
                dict={}
                count=0
                for a in header:
                    dict[a]=count
                    count=count+1
                with open('dictionary_labels_MAGNA.json', 'w') as fp:
                    json.dump(dict, fp)
    print(idx)

elif use_code=='jordi_50':
    output_dict='classes_MagnaTT.json'
    output_file_name = 'gt_classes_MAGNA50.tsv'
    original_file = "annotations_final.csv"
    num_songs = 25863
    num_tags = 188
    top_tags = 50

    tags_matrix = np.zeros((num_songs,num_tags)) + 999 # [OUT] TAGS dimension - only tags
    annotations_matrix = {} # [OUT] ANNOTATIONS dimension - raw matrix
    with open(original_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in csv_reader:
            count=0
            annotations_matrix[csv_reader.line_num-1] = row[:]
            if csv_reader.line_num-1 != 0:
                for r in row[1:-1]:
                    tags_matrix[int(csv_reader.line_num-2)][count] = int(r)
                    count=count+1
            print(str(csv_reader.line_num-1)+'/'+str(num_songs))

    total_counts = np.sum(tags_matrix, axis=0)
    tag_args = total_counts.argsort()[::-1]
    top_idx=tag_args[0:top_tags]

    # map conclusions from TAGS dimension to ANNOTATIONS dimension
    top_annotations_matrix={}
    count=0
    fw = open(output_file_name,"w")
    for row in annotations_matrix:
        tmp=[]
        tmp.append(annotations_matrix[row][0])
        for idx,val in enumerate(top_idx):
            tmp.append(annotations_matrix[row][val+1])
        top_annotations_matrix[count]=tmp
        classes = json.dumps(tmp[1:])
        id = str(tmp[0])
        fw.write("%s\t%s\n" % (id,classes))
        count=count+1

    count=0
    header=top_annotations_matrix[0]
    dict={}
    count=0
    for a in header:
        dict[a]=count
        count=count+1
    with open(output_dict, 'w') as fp:
        json.dump(dict, fp)
