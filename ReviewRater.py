#
# ESOF 2918 Technical Project
#
# Review Rater Project - analyzes sentiment in reviews
#
# Nicholas Imperius, Kristopher Pouling, Jimmy Tsang
#

import nltk

text = "Not one thing in this book seemed an obvious original thought. However, the clarity with which this author explains how innovation happens is remarkable.\n\nAlan Gregerman discusses the meaning of human interactions and the kinds of situations that tend to inspire original and/or clear thinking that leads to innovation. These things include how people communicate in certain situations such as when they are outside of their normal patterns.\n\nGregerman identifies the ingredients that make innovation more likely. This includes people being compelled to interact when they normally wouldn't, leading to serendipity. Sometimes the phenomenon will occur through collaboration, and sometimes by chance such as when an individual is away from home on travel.\n\nI recommend this book for its common sense, its truth and the apparent mastery of the subject by the author."

stopwords = nltk.corpus.stopwords.words("english")

tokenedWords = nltk.word_tokenize(text)

words = []

for word in tokenedWords:
    if word.isalpha():
        if word.lower() not in stopwords:
            words.append(word.lower())
            
print(words)
    
