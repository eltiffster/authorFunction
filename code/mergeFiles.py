#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os

sourcePath = r"/home/Tiffany/ENGL598/samples/oneString/clean3/"

with open(sourcePath + 'merged3.txt', 'w') as outfile:
    for file in os.listdir(sourcePath):
        with open(sourcePath + file) as infile:
            for line in infile:
                outfile.write(line)