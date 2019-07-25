import pandas as pd

df_items = pd.read_table("annotations_final.csv")
fw = open("index_MAGNA.tsv","w") # where to save the file?
for index, row in df_items.iterrows():
	fw.write("%s\t%s\n" % (row['clip_id'],row['mp3_path']))
print('Done!')
