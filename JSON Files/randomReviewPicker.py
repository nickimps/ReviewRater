import json
import random

# Max ID for star1.json is: 59627
# Max ID for star2.json is: 20734
# Max ID for star3.json is: 30652
# Max ID for star4.json is: 75476
# Max ID for star5.json is: 416288

# adds 3 new fields to the json to help with testing
# - originalRating : the rating the reviewer gave
# - ourRating : the rating we assigned to the reviews
# - modelRating : the rating that our system gives to the review

print('Getting random reviews...')

allReviews = []

randNums = []
randNums = random.sample(range(59627), 200)
with open('star1.json') as data_file:
    star1Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
    for element in star1Data:
        if element['ID'] in randNums:
            allReviews.append(element)

randNums = []
randNums = random.sample(range(20734), 200)
with open('star2.json') as data_file:
    star2Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
    for element in star2Data:
        if element['ID'] in randNums:
            allReviews.append(element)

randNums = []
randNums = random.sample(range(30652), 200)
with open('star3.json') as data_file:
    star3Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
    for element in star3Data:
        if element['ID'] in randNums:
            allReviews.append(element)

randNums = []
randNums = random.sample(range(75476), 200)
with open('star4.json') as data_file:
    star4Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
    for element in star4Data:
        if element['ID'] in randNums:
            allReviews.append(element)

randNums = []
randNums = random.sample(range(416288), 200)
with open('star5.json') as data_file:
    star5Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
    for element in star5Data:
        if element['ID'] in randNums:
            allReviews.append(element)

print("Number of Reviews Collected: " + str(len(allReviews)))

random.shuffle(allReviews)

print('\nSaving reviews...')

with open('choosenReviews.json', 'w') as data_file:
    for element in allReviews:
        element.pop('ID', None)
        element['originalRating'] = element.pop('overall')
        element['ourRating'] = ""
        element['modelRating'] = ""
        element = json.dump(element, data_file)
        data_file.write('\n')

print('Reviews Saved.')