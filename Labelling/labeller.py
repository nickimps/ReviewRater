import os
import shutil

## you want your labeller.py (this file) in the same directory as the batches
## for me that is C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\

## change this to the folder that the folder of .txt are saved too, adjust batch1 to batchX depending
## on the one that you have.
batchName = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch1'

## location of the one, two, three... folders
one = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\labelled\\one'
two = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\labelled\\two'
three = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\labelled\\three'
four = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\labelled\\four'
five = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\labelled\\five'

## to run, it will prompt a review, type in either one, two, three, four or five as 
## that will dictate which folder it will be moved to
for file in os.listdir(batchName):
    if file.endswith('.txt'):
        file_path = f"{batchName}\\{file}"
        with open(file_path, 'r') as data_file:
            print('\n' + data_file.read())
            folder = input("Folder: ")
    if folder == 'one':
        shutil.move(file_path, one)
    elif folder == 'two':
        shutil.move(file_path, two)
    elif folder == 'three':
        shutil.move(file_path, three)
    elif folder == 'four':
        shutil.move(file_path, four)
    elif folder == 'five':
        shutil.move(file_path, five)
    else:
        print("you didn't enter valid input, go back later")