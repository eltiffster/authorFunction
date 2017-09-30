#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re

#Set file paths and variables for moving later IN BASH OR TERMINAL:
sourcePath = r"/home/Tiffany/ENGL598/samples/stripped/" #replace with file path to folder with source files
destPath = r"/home/Tiffany/ENGL598/samples/test2/" #replace with file path to where you want
														# clean files to be saved

#IN WINDOWS CMD:
#sourcePath = r"C:/Users/Tiff/Documents/UVic/Classes/ENGL598/samples/stripped/"
#destPath = r"C:/Users/Tiff/Documents/UVic/Classes/ENGL598/samples/oneString/"
fileName = r"" #to be used later

#If you only want a specific set of files in your source folder:
listOfFiles = ['babylonVol1.txt', 'babylonVol2.txt', 'babylonVol3.txt', 'beckoningHand.txt'] #list file names (separated by commas) in the square brackets

for file in listOfFiles:
		#OR
#Iterate over each file in the directory
#for file in os.listdir(sourcePath):

	fileName = file
	#Open the stripped .txt file & read it
	contents = open(sourcePath + fileName, 'r+')
	fulltext = contents.read()

	'''Split the file into lines according to paragraph breaks 
	(as they appear in the raw text). Returns a list of lines.'''
	listLines = fulltext.split('\n\n+')

	'''Define a function to filter out lines we don't want'''
	def filterLines():
		'''Filter out lines that have 2+ capital letters that are
		not followed by an alpha character (e.g. chapter headings
		IN ALL CAPS)'''
		allCaps = re.compile(r'[A-Z][A-Z][^\w]')
		filter1 = filter(lambda i: allCaps.search(i), listLines)
		'''Filter out lines that contain only one capital letter
		in a word boundary (e.g. I(.) at the start of a chapter)'''
		chapNum = re.compile(r'\W*\b\w{0,1}\b')
		filter2 = filter(lambda i: chapNum.search(i), filter1)
		#Filter out Illustration lines
		illus = re.compile(r'\[Illustration:')
		filter3 = filter(lambda i: illus.search(i), filter2)
		#Filter out lines that have asterisks
		ast = re.compile(r'\*')
		filter4 = filter(lambda i: not ast.search(i), filter3)
		return filter4

	toReplace = '\n\n'.join(filterLines())#Call function, pass results to toReplace

#Define a function to change or delete unwanted characters
	def subChars():
		global toReplace
		toReplace = toReplace.replace('_', '')
		toReplace = toReplace.replace('=', '')
		toReplace = toReplace.replace('|', '')
		toReplace = re.sub(r'&lsquo;', '\'', toReplace)
		toReplace = re.sub(r'-{2,}', '—', toReplace)
		return toReplace

	result = subChars()

	#with open('/home/Tiffany/ENGL598/samples/test/' + fileName, 'w') as f:
	f = open(destPath + fileName, 'w')
	f.write(result)
	f.close()