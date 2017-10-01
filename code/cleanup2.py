#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re

#Set file paths and variables for moving later. In bash:
sourcePath = r"../corpus/stripped/"
#destPath = r"/path to folder/foldername/"

#In Windows cmd:
#sourcePath = r"C:/Users/Tiff/Documents/UVic/Classes/ENGL598/samples/stripped/"
#destPath = r"C:/Users/Tiff/Documents/UVic/Classes/ENGL598/samples/oneString/"
fileName = r""
#listOfFiles = ['']
#Clean specific file(s) instead of all items in a folder
#for file in listOfFiles:

#Iterate over each file in the directory
for file in os.listdir(sourcePath):
    fileName = file
    print(fileName)

    contents = open(sourcePath + fileName, 'r+') #Open the stripped .txt file & read it
    fulltext = contents.read()
    '''Split the file into lines according to line breaks 
    (as they appear in the raw text). Returns a list of lines.'''
    listLines = fulltext.splitlines()

    '''Define a function to filter out lines we don't want 
    (e.g. chapter headings which are ALL CAPs)'''
    def filterLines():
        #Filter out all lines with 2+ capital letters
        allCaps = re.compile('[A-Z][A-Z]')
        filtered = filter(lambda i: not allCaps.search(i), listLines)
        '''Filter out all lines that do not contain more than 3 or more
        alpha characters'''
        Words = re.compile(r'\b[A-Za-z]{3,}\b')
        newFiltered = filter(lambda i: Words.search(i), filtered)
        return newFiltered

    listLines = filterLines()

    '''Define a function mergeHyp to find words that are split 
    by a hyphen across a line break and merge them back together'''
    def mergeHyp():
        for line in listLines:
            line.decode('utf-8')
            if line.endswith('- ') == True: #Check if line ends in hyphen            
                #print(line)
                index = listLines.index(line)#Find the line's place in the list           
                newLine = line.replace('- ', '') #Remove the hyphen
                #print(newLine)
                listLines.remove(listLines[index])
                #print(listLines)
                #print(newLine)
                listLines.insert(index, newLine)#Update list by inserting newLine
                seq = [str(listLines[index]), str(listLines[index+1])]
                listLines.remove(listLines[index])
                listLines.insert(index, ''.join(seq))#Update list by inserting newLine
                del(listLines[index+1])
                #print(listLines)
        return listLines

    merged = mergeHyp()
    #print(merged)

    '''Generator to merge all lines into a single string,
     returns that string'''
    global toReplace
    def linesGen():
        oneString = []
        listGen = list(range(len(merged)))#calc total # of lines (x), create a list of x entries
        for i in listGen[:-1]: #skip the last iteration tp prevent out of range error message
            if i > 0 and i % 2 != 0: #skip the first line and every other line to avoid repetition
                currTxt = str.strip(merged[i]) #define two sequential strings
                prevTxt = str.strip(merged[i - 1])
                seq = [str(prevTxt), str(currTxt)] #merge the two strings into one
                oneString.append(' '.join(seq)) #make one big list, each entry is a string of the above
        oneString.append(' ' + merged[-1]) #add the last item (that we skipped earlier) back in
        oneString = ' '.join(oneString) #return one looooooooong string to rule them all
        return oneString

    toReplace = linesGen()
    #print(toReplace)

    #Replace common OCR errors with the correct chracters
    def replaceChars():
        global toReplace

        #turn l< and i< into k
        a = re.compile('l<|i<')
        toReplace = a.sub('k', toReplace)

        #turn 3', }', 3^, }^, j'^ into y
        b = re.compile(r'3\'|\}\'|3\^|\}\^|j\'|j\^|j\/|\^\'|\}\*')
        toReplace = b.sub('y', toReplace)

        #Turn 'mc' or 'Mc' into me and Me respectively
        toReplace = re.sub(r'mc([ ?.!,;:])', r'me\1', toReplace)
        toReplace = re.sub(r'Mc ', r'Me ', toReplace)

        # '( )' → O
        toReplace = toReplace.replace('( )', 'O')    

        #'v/', 'v^', 'zv' → 'w'
        c = re.compile('v/|v^|zv|\'\'|\\/|\\v|vv')
        toReplace = c.sub('w', toReplace)

        #'liis', 'Jis', and 'ii' → 'his'
        d = re.compile('(?=^t)li|(?=^T)li|li(?=is)|Ji|(?<![lJ])ii|]i')
        toReplace = d.sub('h', toReplace)

        #Turn 1 into ! if it is before ' or "", or if it is before a capital letter.
        toReplace = re.sub(' 1 ([A-Z\"\'])', r' ! \1', toReplace)

        # l) and ]) → b
        toReplace = re.sub(r'[l\]]\)', r'b', toReplace)

        #' 1 ', ' / ' → ' I '
        e = re.compile(' 1 | / ')
        toReplace = e.sub(' I ', toReplace)

        #'tLi' or 'tli' → 'th'
        f = re.compile('tLi|tli')
        toReplace = f.sub('th', toReplace)

        g = re.compile('httle|Httle')
        toReplace = g.sub('little', toReplace)

        h = re.compile('Euss|Kuss')
        toReplace = h.sub('Russ', toReplace)

        i = re.compile(r'j;|i;|\*;')
        toReplace = i.sub('g', toReplace)

        j = re.compile(r'\*\*|\*\\|\*\'|\'\*|\^\*|\'\'')
        toReplace = j.sub('"', toReplace)

        k = re.compile(r'j\)|\^i|\]:|i\)|\}\)')
        toReplace = k.sub('p', toReplace)

        #Punctuation * letter. E.g. cried. \* Such, cried. \'Such → cried. ' Such
        toReplace = re.sub(r'([?.!,;: ]) \\[\'\*] ([a-zA-Z])', r'\1 \' \2' , toReplace)

        #Some other convenient replacement rules
        toReplace = toReplace.replace('*-', 't')
        toReplace = toReplace.replace(' ol ', ' of ')
        toReplace = toReplace.replace('agamst', 'against')
        toReplace = toReplace.replace('tiling', 'thing')
        toReplace = toReplace.replace(' sec ', ' see ')
        toReplace = toReplace.replace(' arc ', ' are ')
        toReplace = toReplace.replace('i\'', 'r')
        toReplace = toReplace.replace('hfe', 'life')
        toReplace = toReplace.replace('yoii^', 'your')
        toReplace = toReplace.replace('Eome', 'Rome')
        toReplace = toReplace.replace('.^', 's')
        toReplace = toReplace.replace('" *', '" \'')
        toReplace = toReplace.replace('* \"', '\' \"')
        toReplace = toReplace.replace(' _I_ ', 'I')
        toReplace = toReplace.replace('Iliram', 'Hiram')
        toReplace = toReplace.replace('0h', 'Oh')
        toReplace = toReplace.replace('::', 'x')
        toReplace = toReplace.replace('hke', 'like')
        toReplace = toReplace.replace('\\V', 'W')
        toReplace = toReplace.replace('(j', 'q')
        toReplace = toReplace.replace('s^gainst', 'against')
        toReplace = toReplace.replace('/', ',\'')
        toReplace = toReplace.replace(' * ', '\'')
        toReplace = toReplace.replace('_', '')
        toReplace = toReplace.replace('=', '')
        toReplace = toReplace.replace('/\'', '"')

        #Get rid of lingering hyphens between words
        toReplace = re.sub(r'[a-z]- [a-z]','', toReplace)
        
        #Turn lingering asterisks into single quotes
        toReplace = re.sub(r'\*','\'', toReplace)

        return toReplace

    replaced = replaceChars()
    #print(replaced)

    #Define a function to normalize punctuation
    def normPunct():
        global replaced
        '''Turn punct sandwiched by spaces into punct 
        with one trailing space (except dashes and hyphens)'''
        replaced = re.sub(r'\s([?.!,;:()](?:\s|$))', r'\1', replaced)

        #If punct is sandwiched by letters, add a space after it
        replaced = re.sub(r'([a-zA-Z][?.!,;:()"])([a-zA-Z])', r'\1 \2', replaced)

        #Fix quotation mark formatting and spacing issues
        #replaced = replaced.replace('" ', '"') #e.g." You..." → "You..."
        replaced = re.sub(r'([?.!";:()]) "[^A-Za-z]', r'\1" ', replaced) #e.g."...! " people say → "...!" people say
        replaced = re.sub(r'([?.!";:])"([A-Z])', r'\1" \2', replaced) #e.g. ...them."It... → them. "It...
        replaced = re.sub(r'([?.!";:])"" ([A-Z])', r'\1" "\2', replaced) #e.g. said."" But... → said." "But...
        #Do the same for single quotes
        replaced = re.sub(r'([?.!";:()]) \'[^A-Za-z]', r'\1\' ', replaced) #e.g. ...! ' people say → ...!' people say
        replaced = re.sub(r'([?.!";:])\'([A-Z])', r'\1\' \2', replaced) #e.g. ...them."It... → them. "It...
        replaced = re.sub(r'([?.!";:])\'\' ([A-Z])', r'\1\' \'\2', replaced) #e.g

        #Replace 2 or more spaces in a row with one space
        replaced = re.sub(' +', ' ', replaced)

        return replaced

    result = normPunct()
    #print(result)
    f = open('/home/Tiffany/ENGL598/samples/oneString/clean3/' + fileName, 'w')
    f.write(result)
    f.close

    #Move the resulting .txt file from directory stripped to oneString
    #shutil.move(sourcePath + fileName, destPath + fileName)