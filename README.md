# ReviewRater

## Summary of Project

   The purpose of this project is to research and create a prototype that can take in a product
review, analyze it, and understand the sentiment behind it to determine whether it is a positive or
a negative review. This project will provide a deeper understanding of the feeling a consumer has
towards a product by using sentiment analysis to determine whether or not a consumer is
satisfied with their purchase. This information can then be later used as feedback to make
adjustments or improvements to a product or to gain an advantage when marketing to certain
audiences. This program would have many uses, such as in a retail environment, specifically an
internet platform, as it would improve how well a product has been reviewed and will give other
consumers a more in-depth, accurate rating.

   For example, a review on a product is provided that consists of points as to why the
customer is satisfied with the product but because the packaging was damaged, the consumer
gave it a negative review. The program would then take the review and perform sentiment
analysis using natural language processing to determine whether the consumer found the product
satisfactory and provide the review with an accurate number rating.

## Installation

Install spaCy, version: 2.3.5

```shell
pip install -U spacy==2.3.5
```
Install the English Model for spaCy, version: 2.3.0

```shell
pip install -U https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz
```

## Usage

The Following will Train and Test the TEST_REVIEW review
```shell
python ReviewRater.py train
```
The Following will just Test the TEST_REVIEW review
```shell
python ReviewRater.py test
```

# To Label Reviews For Training
The batches.zip contains 25 batches each containing **200 reviews**, in .txt files.<br>
The labelled.zip is a folder that contains a hierarchy of folders that will store the rating given to each one, there is a folder called one, two, etc..<br>
1. Download and unzip the batches.zip and the labelled.zip file into a folder.<br>
2. Download the labeller.py into the same folder.<br>
3. To run the labeller.py program: <br>
_NOTE: you will need to change the file path in labeller.py to the current location on your PC, let me know if it still isnt working_
```shell
python labeller.py
```
You will be prompted with a review and input. For the input enter the number of the rating you wish to give it, this will move it from the batch folder to the given folder in the labelled hierarchy. You can stop the program (CTRL + C) and resume later, it will not restart all the reviews - just whatever you have left.<br>

## Batch Tracker
- [x] batch1
- [ ] batch2
- [ ] batch3
- [ ] batch4
- [ ] batch5
- [ ] batch6
- [ ] batch7
- [ ] batch8
- [ ] batch9
- [ ] batch10
- [ ] batch11
- [ ] batch12
- [ ] batch13
- [ ] batch14
- [ ] batch15
- [ ] batch16
- [ ] batch17
- [ ] batch18
- [ ] batch19
- [ ] batch20
- [ ] batch21
- [ ] batch22
- [ ] batch23
- [ ] batch24
- [ ] batch25

