import pandas as pd

def merge_chunks(sets):
	merged = {}
	all_items = list(sets.items())
	for i, (key, val) in enumerate(all_items):
		nearest = None
		max_common = 0
		for j in range(i+1, len(all_items)):
			ikey = all_items[j][0]
			ival = all_items[j][1]
			common = len(val.intersection(ival))
			if common > max_common:
				nearest = ikey
				max_common = common
		if max_common > 0:
			new_key = set(key).union(nearest)
			# print(key, new_key)
			merged[tuple(list(new_key))] = val.union(sets[nearest])
		else:
			merged[key] = val
	return merged

df = pd.read_csv('face_clusters_small.csv')
# df.drop('embedding', axis=1, inplace=True)
# df.to_csv('face_clusters_small.csv')

clc = dict()
for chunk, cluster in zip(df.chunk, df.cluster):
	chunk = tuple((chunk,))
	if chunk in clc:
		clc[chunk].add(cluster)
	else:
		clc[chunk] = set([cluster])

for key, val in clc.items():
	print('{}: {}'.format(key, val))

merged = clc
for i in range(50):
	print('\n\n')
	merged = merge_chunks(merged)
	for key, val in merged.items():
		print('{}: {}'.format(key, val))

