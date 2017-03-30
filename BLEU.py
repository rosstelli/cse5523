import read as d
import sample as sa
import nltk
import numpy


filename = 'questions.csv'
#filename = 'medium.csv'
sam_size = 100000 # Grabs this many positive and negative sample points randomly


def runBLEU(field, data):
	# Types of characters to ignore
	ignore = ';:.\'",?<>-=_+[]{}'

	# Output file for BLEU run
	#output = open('outputNeg.out', 'w')
	#outputp = open('outputPos.out', 'w')

	l = len(data)
	x = [0] * l 
	i = 0

	#Store error for each kind of 
	err = [0] * 101

	for line in data:
		q1 = line[field[3]].translate(None, ignore).split()
		q2 = line[field[4]].translate(None, ignore).split()

		BLEUscore = nltk.translate.bleu_score.sentence_bleu([q1], q2)
		x[i] = BLEUscore

		label = (int(float(line[field[5]])))
		
		for j in range(0,101):
			threshold = float(j) / 100
			# If the BLEUscore is higher than the threshold, then we assume that
			# the questions are a match
			classified = int(BLEUscore > threshold)
			
			if (label != classified):
#				print 'error at:' + str(line[field[5]]) + ', ' + str(classified) + ', ' + str(BLEUscore) + ', thr=' + str(threshold)
				err[j] = err[j] + 1
#			else:
#				print 'no-error at:' + str(line[field[5]]) + ', ' + str(classified) + ', ' + str(BLEUscore) + ', thr=' + str(threshold)
		i = i + 1


	return x, err

"""
def calculateError(threshold, bleuscores, data, field):
	error = 0
	i = 0
	for line in data:
		if (int(bleuscores[0] - threshold + 1) != int(float(line[field[5]]))):
			error = error + 1
		i = i + 1

	return error
"""

field, data = d.readData(filename);
print 'Loaded data'
sampled = sa.sample50_50(data, field, sam_size)
print 'Sampled data'
scores, err = runBLEU(field, sampled)
for i in range(0, len(err)):
	print str(i) + ',' + str(float(err[i]) / (2 * sam_size))


