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
