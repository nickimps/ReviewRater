import json
import random
import os

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
random.shuffle(star5Data)
random.shuffle(star3Data)
random.shuffle(star2Data)
random.shuffle(star4Data)
random.shuffle(star1Data)

print('\nSaving reviews...')

"""
# to create the batches for us to review
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch1'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch2'
name3 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch3'
name4 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch4'
name5 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch5'
for element in allReviews:
    if num <= 200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 400:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 600:
        with open(os.path.join(name3, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 800:
        with open(os.path.join(name4, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 1000:
        with open(os.path.join(name5, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
"""

"""
# this creates a folder for each given star rating
print("Creating one....")
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\train\\one\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\test\\one\\'
for element in star1Data:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
print("One done.")

print("Creating two....")
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\train\\two\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\test\\two\\'
for element in star2Data:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
print("Two done.")

print("Creating three....")
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\train\\three\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\test\\three\\'
for element in star3Data:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
print("Three done.")

print("Creating four....")
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\train\\four\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\test\\four\\'
for element in star4Data:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
print("Four done.")

print("Creating five....")
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\train\\five\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\dataset\\test\\five\\'
for element in star5Data:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', '""')
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
print("Five done.")
"""

# this created the pos, neut and neg folders
"""
neutReviews = []
for element in star2Data:
    neutReviews.append(element)
for element in star3Data:
    neutReviews.append(element)
for element in star4Data:
    neutReviews.append(element)
random.shuffle(neutReviews)


num = 1

name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\labelledReviews\\train\\pos\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\labelledReviews\\test\\pos\\'
for element in star5Data:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', None)
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', None)
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
        
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\labelledReviews\\train\\neg\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\labelledReviews\\test\\neg\\'
for element in star1Data:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', None)
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', None)
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break

num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\labelledReviews\\train\\neut\\'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\labelledReviews\\test\\neut\\'
for element in neutReviews:
    if num < 200:
        with open(os.path.join(name2, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', None)
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num < 2200:
        with open(os.path.join(name, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', None)
            while True:
                if review is None:
                    review = element.pop('reviewText', None)
                else:
                    break
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
"""
    
# This one creates the choosenReviews file    
"""
with open('choosenReviews.json', 'w') as data_file:
    for element in allReviews:
        element.pop('ID', None)
        element['originalRating'] = element.pop('overall')
        element['ourRating'] = ""
        element['modelRating'] = ""
        element = json.dump(element, data_file)
        data_file.write('\n')
"""

print('Reviews Saved.')