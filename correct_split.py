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


def cheapest_chunk(clc, split_list):
	split_actors = set()
	all_chunks = set(clc.keys())
	candidate_chunks = all_chunks.difference(split_list)

	for chunk in split_list:
		split_actors = split_actors.union(clc[chunk])

	the_chunk = None

	min_common_actors = 1000
	for chunk in candidate_chunks:
		candidate_split_list = split_list[:]
		candidate_split_list.append(chunk)
		n_common_actors = len(split_intersection(clc, candidate_split_list))
		if n_common_actors < min_common_actors:
			min_common_actors = n_common_actors
			the_chunk = chunk
		
	common_actors = None
	common_actors = split_actors.intersection(clc[the_chunk])
	
	return the_chunk, common_actors


def closest_chunk(clc, split_list):
	split_actors = set()
	all_chunks = set(clc.keys())
	candidate_chunks = all_chunks.difference(split_list)

	for chunk in split_list:
		split_actors = split_actors.union(clc[chunk])

	the_chunk = None
	max_common_actors = 0
	for chunk in candidate_chunks:
		n_common_actors = len(split_actors.intersection(clc[chunk]))
		if n_common_actors > max_common_actors:
			max_common_actors = n_common_actors
			the_chunk = chunk
	
	if the_chunk is None:
		clcc = dict(clc)
		for chunk in split_list:
			del clcc[chunk]
		sorted_clcc = sorted(clcc.items(), key=lambda t: len(t[1]))
		the_chunk = sorted_clcc[0][0]
	
	common_actors = None
	if the_chunk is not None:
		common_actors = split_actors.intersection(clc[the_chunk])
	
	return the_chunk, common_actors
	

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
	val_list = range(41, 50)
	common_actors = split_intersection(clc, val_list)
	print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))
	
	val_list = [0, 1, 12, 32, 42]
	candidate_chunk, candidate_common = closest_chunk(clc, val_list)
	print('\nClosest chunk for {} is {} with common actors: {}\n'.format(val_list, candidate_chunk, candidate_common))

	val_list.append(candidate_chunk)
	common_actors = split_intersection(clc, val_list)
	print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))

	val_list = [0]
	print('\n\n*** Using closest chunk.***\n')
	for i in range(10):
		print('Step {}'.format(i))
		candidate_chunk, candidate_common = closest_chunk(clc, val_list)
		print('\nClosest chunk for {} is {} with common actors: {}\n'.format(val_list, candidate_chunk, candidate_common))
		val_list.append(candidate_chunk)
		common_actors = split_intersection(clc, val_list)
		print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))

	val_list = [0]
	print('\n\n*** Using cheapest chunk.***\n')
	for i in range(10):
		print('Step {}'.format(i))
		candidate_chunk, candidate_common = cheapest_chunk(clc, val_list)
		print('\nCheapest chunk for {} is {} with common actors: {}\n'.format(val_list, candidate_chunk, candidate_common))
		val_list.append(candidate_chunk)
		common_actors = split_intersection(clc, val_list)
		print('\nLeaked actors for {} : {}\n'.format(val_list, common_actors))


	# merged = dict()
	# for chunk, cluster in zip(df.chunk, df.cluster):
	# 	chunk = tuple((chunk,))
	# 	if chunk in merged:
	# 		merged[chunk].add(cluster)
	# 	else:
	# 		merged[chunk] = set([cluster])

	# for i in range(args.n):
	# 	merged = merge_chunks(merged)

	# for key, val in merged.items():
	# 	print('{}: {}'.format(key, val))

