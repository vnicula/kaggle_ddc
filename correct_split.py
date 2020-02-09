import argparse
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


def split_intersection(clc, split_list):
	train_actors = set()
	common_actors = set()
	all_chunks = set(clc.keys())
	chunks1 = all_chunks.difference(split_list)
	for chunk in chunks1:
		train_actors = train_actors.union(clc[chunk])
	# print('Train actors: {}'.format(train_actors))
	for chunk in split_list:
		common_actors = common_actors.union(clc[chunk].intersection(train_actors))

	return common_actors
	

if __name__ == '__main__':

	argp = argparse.ArgumentParser()
	argp.add_argument('--n', type=int, default=20)
	args = argp.parse_args()

	df = pd.read_csv('face_clusters_small.csv')
	# df.drop('embedding', axis=1, inplace=True)
	# df.to_csv('face_clusters_small.csv')

	clc = dict()
	for chunk, cluster in zip(df.chunk, df.cluster):
		if chunk in clc:
			clc[chunk].add(cluster)
		else:
			clc[chunk] = set([cluster])

	for key, val in clc.items():
		print('{}: {}'.format(key, val))

	val_list = [0, 1, 12, 32, 42]
	common_actors = split_intersection(clc, val_list)
	print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))
	val_list = [0, 1, 8, 12, 32, 42]
	common_actors = split_intersection(clc, val_list)
	print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))
	val_list = [0, 1, 12, 17, 32, 42]
	common_actors = split_intersection(clc, val_list)
	print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))
	val_list = [0, 1, 12, 22, 32, 42]
	common_actors = split_intersection(clc, val_list)
	print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))
	

	merged = dict()
	for chunk, cluster in zip(df.chunk, df.cluster):
		chunk = tuple((chunk,))
		if chunk in merged:
			merged[chunk].add(cluster)
		else:
			merged[chunk] = set([cluster])

	for i in range(args.n):
		merged = merge_chunks(merged)

	for key, val in merged.items():
		print('{}: {}'.format(key, val))

