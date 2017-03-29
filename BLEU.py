import read as d
import nltk
import numpy


filename = 'questions.csv'


def runBLEU(field, data):
	# Cumulative error
	error = 0

	# Types of characters to ignore
	ignore = ';:.\'",?<>-=_+[]{}'

	# Output file for BLEU run
	#output = open('outputBLEU' + str(threshold) + '.out', 'w')

	l = len(data)
	x = [0] * l 
	i = 0
	err = [0] * 100
	for line in data:
		var1 = line[field[3]].translate(None, ignore).split()
		var2 = line[field[4]].translate(None, ignore).split()

		#print line[field[3]]
		BLEUscore = nltk.translate.bleu_score.sentence_bleu([var1], var2)
		x[i] = BLEUscore

		for j in range(0,100):
			threshold = float(j) / 100
			classified = int(BLEUscore > threshold)
			if ((int(float(line[field[5]])) != classified)):
				err[j] = err[j] + 1
		i = i + 1
#		classified = int(BLEUscore - threshold + 1)
#
#		#print int(float(line[field[5]])) == classified
#		if ((int(float(line[field[5]])) != classified)):
#			error = error + 1
#
#	print 'Error = ' + str(float(error) / len(data)) + ' for threshold ' + str(threshold)
#	output.write('Error rate:\n' + str(float(error) / len(data)))
#	output.write('\n\nAccuracy:\n' + str(1 - float(error) / len(data)))
#	output.close()
	print 'Error'
	print err
	print
	print 'Max Bleu Score'
	print max(x)

	print numpy.mean(x)
	return x

def calculateError(threshold, bleuscores, data, field):
	error = 0
	i = 0
	for line in data:
		if (int(bleuscores[0] - threshold + 1) != int(float(line[field[5]]))):
			error = error + 1
		i = i + 1

	return error



#Threshold, everything above will be classified as 1 and below as 0
threshold = 0

field, data = d.readData(filename);
scores = runBLEU(field, data)
#print scores
# field[3] is question1, field[4] is question2, field[5] is the class
#for i in range(0,100):
#	threshold = float(i) / 100

x = [0] * 100

print 'Calculating error'

for i in range(0,100):
	threshold = float(i) / 100
	x[i] = calculateError(threshold, scores, data, field)
	print i
	
print x












