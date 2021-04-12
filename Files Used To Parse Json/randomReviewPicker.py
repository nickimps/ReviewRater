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
randNums = random.sample(range(59627), 10000)
with open('star1.json') as data_file:
    star1Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
for element in star1Data:
    if element['ID'] in randNums:
        allReviews.append(element)
print("Gathered 1 Star")

randNums = []
randNums = random.sample(range(20734), 10000)
with open('star2.json') as data_file:
    star2Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
for element in star2Data:
    if element['ID'] in randNums:
        allReviews.append(element)
print("Gathered 2 Star")

randNums = []
randNums = random.sample(range(30652), 10000)
with open('star3.json') as data_file:
    star3Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
for element in star3Data:
    if element['ID'] in randNums:
        allReviews.append(element)
print("Gathered 3 Star")

randNums = []
randNums = random.sample(range(75476), 10000)
with open('star4.json') as data_file:
    star4Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
for element in star4Data:
    if element['ID'] in randNums:
        allReviews.append(element)
print("Gathered 4 Star")

randNums = []
randNums = random.sample(range(416288), 10000)
with open('star5.json') as data_file:
    star5Data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")
for element in star5Data:
    if element['ID'] in randNums:
        allReviews.append(element)
print("Gathered 5 Star")

print("Number of Reviews Collected: " + str(len(allReviews)))

random.shuffle(allReviews)
random.shuffle(star5Data)
random.shuffle(star3Data)
random.shuffle(star2Data)
random.shuffle(star4Data)
random.shuffle(star1Data)

print('\nSaving reviews...')




# Creating the folder system for the Reviewer Labelled Testing
name1tr = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\train\\one'
name2tr = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\train\\two'
name3tr = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\train\\three'
name4tr = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\train\\four'
name5tr = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\train\\five'
name1te = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\one'
name2te = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\two'
name3te = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\three'
name4te = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\four'
name5te = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\test\\five'

num = 1
for element in star1Data:
    if num <= 5000:
        with open(os.path.join(name1tr, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 10000:
        with open(os.path.join(name1te, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break            
print("Done 1 Star")
            
num = 1
for element in star2Data:
    if num <= 5000:
        with open(os.path.join(name2tr, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 10000:
        with open(os.path.join(name2te, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break            
print("Done 2 Star")

num = 1
for element in star3Data:
    if num <= 5000:
        with open(os.path.join(name3tr, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 10000:
        with open(os.path.join(name3te, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break            
print("Done 3 Star")
            
num = 1
for element in star4Data:
    if num <= 5000:
        with open(os.path.join(name4tr, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 10000:
        with open(os.path.join(name4te, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
print("Done 4 Star")
            
num = 1
for element in star5Data:
    if num <= 5000:
        with open(os.path.join(name5tr, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 10000:
        with open(os.path.join(name5te, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    else:
        break
print("Done 5 Star")







""" # This created all the batchs for us
# to create the batches for us to review and test
num = 1
name = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch1'
name2 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch2'
name3 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch3'
name4 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch4'
name5 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch5'
name6 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch6'
name7 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch7'
name8 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch8'
name9 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch9'
name10 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch10'
name11 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch11'
name12 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch12'
name13 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch13'
name14 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch14'
name15 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch15'
name16 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch16'
name17 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch17'
name18 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch18'
name19 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch19'
name20 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch20'
name21 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch21'
name22 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch22'
name23 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch23'
name24 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch24'
name25 = 'C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Personally Labelled\\batch25'

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
    elif num <= 1200:
        with open(os.path.join(name6, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 1400:
        with open(os.path.join(name7, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 1600:
        with open(os.path.join(name8, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 1800:
        with open(os.path.join(name9, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 2000:
        with open(os.path.join(name10, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 2200:
        with open(os.path.join(name11, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 2400:
        with open(os.path.join(name12, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 2600:
        with open(os.path.join(name13, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 2800:
        with open(os.path.join(name14, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 3000:
        with open(os.path.join(name15, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 3200: ### 16
        with open(os.path.join(name16, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 3400:
        with open(os.path.join(name17, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 3600:
        with open(os.path.join(name18, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 3800:
        with open(os.path.join(name19, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 4000:
        with open(os.path.join(name20, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 4200:
        with open(os.path.join(name21, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 4400:
        with open(os.path.join(name22, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 4600:
        with open(os.path.join(name23, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 4800:
        with open(os.path.join(name24, str(num) + '.txt'), 'w') as data_file:
            review = element.pop('reviewText', '""')
            if '"' in review:
                review = review.replace('""', '')
            data_file.write(review)
            num += 1
    elif num <= 5000:
        with open(os.path.join(name25, str(num) + '.txt'), 'w') as data_file:
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