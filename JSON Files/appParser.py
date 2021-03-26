# In Appliances.json we have the following titles:
#   - overall       <-- this is the rating they gave
#   - vote
#   - verified
#   - reviewTime
#   - reviewerID
#   - asin
#   - style
#   - reviewerName
#   - reviewText    <-- this is the review
#   - summary       <-- Review Title
#   - unixReviewTime
#   - image

import json

# This is what is in the appliance.json file
# 5: 416288
# 4: 75476
# 3: 30652
# 2: 20734
# 1: 59627
# # of Reviews: 602777



print('Parsing...')


with open('Appliances.json') as data_file:
    data = json.loads("[" + 
        data_file.read().replace("}\n{", "},\n{") + 
    "]")



star5 = []
star4 = []
star3 = []
star2 = []
star1 = []

for element in data:
    element.pop('vote', None)
    element.pop('verified', None)
    element.pop('reviewTime', None)
    element.pop('reviewerID', None)
    element.pop('asin', None)
    element.pop('style', None)
    element.pop('reviewerName', None)
    element.pop('unixReviewTime', None)
    element.pop('image', None)
    if (element['overall'] == 5):
        star5.append(element)
    elif (element['overall'] == 4):
        star4.append(element)
    elif (element['overall'] == 3):
        star3.append(element)
    elif (element['overall'] == 2):
        star2.append(element)
    elif (element['overall'] == 1):
        star1.append(element)
        

# Seperates all reviews into different files each with their own unique ID number   
IDNum = 1
with open('star5.json', 'w') as data_file5:
    for element in star5:
        element['ID'] = IDNum
        IDNum += 1
        element = json.dump(element, data_file5)
        data_file5.write('\n')

IDNum = 1   
with open('star4.json', 'w') as data_file4:
    for element in star4:
        element['ID'] = IDNum
        IDNum += 1
        element = json.dump(element, data_file4)
        data_file4.write('\n')

IDNum = 1   
with open('star3.json', 'w') as data_file3:
    for element in star3:
        element['ID'] = IDNum
        IDNum += 1
        element = json.dump(element, data_file3)
        data_file3.write('\n')

IDNum = 1
with open('star2.json', 'w') as data_file2:
    for element in star2:
        element['ID'] = IDNum
        IDNum += 1
        element = json.dump(element, data_file2)
        data_file2.write('\n')

IDNum = 1
with open('star1.json', 'w') as data_file1:
    for element in star1:
        element['ID'] = IDNum
        IDNum += 1
        element = json.dump(element, data_file1)
        data_file1.write('\n')


print('Parsed.')