import random
import numpy as np

def sample50_50(data, field, sample):
	p = [] # Positive labeled data (matching questions)
	n = [] # 0 labeled data (non-matching questions)
	for l in data:
		label = (int(float(l[field[5]])))
		if (label == 1):
			p.append(l)
		else:
			n.append(l)
	
	random.shuffle(p)
	random.shuffle(n)
	p = np.asarray(p)
	n = np.asarray(n)
	p = p[range(0,sample)]
	n = n[range(0,sample)]
	return np.concatenate((p, n))
#	return p, n
