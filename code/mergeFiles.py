#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os

sourcePath = r"/path to folder/foldername/"

with open(sourcePath + 'merged.txt', 'w') as outfile:
    for file in os.listdir(sourcePath):
        with open(sourcePath + file) as infile:
            for line in infile:
                outfile.write(line)