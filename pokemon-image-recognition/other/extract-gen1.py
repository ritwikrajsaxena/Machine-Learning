# This is used for copying all of the first generation photos into the "Gen 1 Data" folder.
import os
import shutil

# Get the names of all the first-gen pokemon:
firstGenNames = []
with open("../Pokemon Datasets/FirstGenPokemon.csv", 'r') as f:
    for row in f:
        firstGenNames.append(row.split(',')[1])

firstGenNames.pop(0)

# print((firstGenNames))

# Copy the names from the full dataset into a new folder:
rootPath = "/Users/eric/Documents/Classes/CS 5232/Project Data/"
for name in firstGenNames:
    src = rootPath + "Complete Test Set/" + name
    dest = rootPath + "Gen 1 Set/" + name
    destination = shutil.copytree(src, dest)