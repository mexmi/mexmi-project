import os
import os.path as osp
import numpy as np

def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

def clipDataTopX(dataToClip, top=2):
	# res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
	res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
	return np.array(res)

def DataAtIdx(dataToClip, labels):
	# res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
	data=[]
	for i in range(len(labels)):
		s = dataToClip[i]
		l = labels[i]
		data.append([s[l]])
	return np.array(data)