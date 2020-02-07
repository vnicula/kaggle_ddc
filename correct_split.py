import pandas as pd

df = pd.read_csv('face_clusters.csv')
df.drop('embedding', axis=1, inplace=True)
df.to_csv('face_clusters_small.csv')

clc = dict()
for chunk, cluster in zip(df.chunk, df.cluster):
	if cluster in clc:
		clc[cluster].append(chunk)
	else:
		clc[cluster] = [chunk]

changed = True
while changed:

	changed = False
	for cluster, chunks in clc.items():
		chunks_set = set(chunks)

		for icluster, ichunks in clc.items():
			if cluster != icluster and len(chunks_set.intersection(ichunks)) > 0:
				# aggregate values and delete second key
				clc[cluster] = list(set(clc[cluster]).union(ichunks))
				clc[icluster] = []
				changed=True

for key in clc:
	if len(clc[key]) > 0:
		print(key, clc[key])