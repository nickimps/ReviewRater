import os, json

containerArray = []

locationOf1 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\one\\'
locationOf2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\two\\'
locationOf3 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\three\\'
locationOf4 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\four\\'
locationOf5 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\five\\'

for i in range(6):
    insideArray = []
    if i == 0:
        containerArray.append(insideArray)
    else:
        if i == 1:
            location = locationOf1
        elif i == 2:
            location = locationOf2
        elif i == 3:
            location = locationOf3
        elif i == 4:
            location = locationOf4
        else:
            location = locationOf5
            
        for file in os.listdir(location):
            with open(os.path.join(location, file), 'r') as fileText:
                insideArray.append(fileText.read())
        containerArray.append(insideArray)

reviews = {}
with open('originallyLabelledReviews.json', 'w') as data_file:
    for i in range(1,6):
        for review in containerArray[i]:
            json.dump({"rating": i, "review": review}, data_file)
            data_file.write('\n')