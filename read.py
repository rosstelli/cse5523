import csv

def readData(filename):
	csvfile = open(filename, 'r');
	csvread = csv.DictReader(csvfile);
	fields = csvread.fieldnames
	data = [line for line in csvread]
	return fields, data
