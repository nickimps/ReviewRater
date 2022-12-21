#
# ESOF 2918 Technical Project
#
# Review Rater Project - analyzes sentiment in reviews
#
# Nicholas Imperius, Kristopher Poulin, Jimmy Tsang
#
# Followed guide made by Kyle Stratis on https://realpython.com/sentiment-analysis-python/#building-your-own-nlp-sentiment-analyzer
#

import os
import json
import sys
import random
import spacy
from spacy.util import minibatch, compounding
import pandas as pd

##
# Train Model Function
# - trains a textcategorizer model and saves it to the local storage
##

# This is a test


def train_model(training_data: list, test_data: list, iterations: int = 20) -> None:
    # Build the pipeline that we are going to be using
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    # Create labels for categorization
    textcat.add_label("one")
    textcat.add_label("two")
    textcat.add_label("three")
    textcat.add_label("four")
    textcat.add_label("five")

    # Another change is done here

    # Train only textcat model
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Print Statements to let user know we are beginning the training now
        print("Beginning training...")
        print("Loss\t\t\tPrecision\t\tRecall\t\t\tF-score")

        # A creates a lot of random numbers, starts at 4 and multiplies by 1.001 each time until 32 is reached
        batch_sizes = compounding(4.0, 32.0, 1.001)

        # Start the training iterations loop
        for i in range(iterations):
            # Start the training iteration
            print("Training iteration " + str(i))

            # Initialize the loss variable shuffle the training data
            loss = {}
            random.shuffle(training_data)

            # Create the batches of data to test
            batches = minibatch(training_data, size=batch_sizes)

            # For every batch in the batches we want to extract the review and its label and update the model
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                # Call evaluate_model function to compute return the precision, recall, and f-score for this batch
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer, textcat=textcat, test_data=test_data)
                print(str(loss['textcat']) + '\t' + str(evaluation_results['precision']) + '\t' + str(
                    evaluation_results['recall']) + '\t' + str(evaluation_results['f-score']))

    # Save the model so it can be used for testing
    modelName = "ManuallyLabelledModel_3"  # MODEL NAME ##
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(modelName)
    print("Model: " + modelName + " has been saved.")

##
# Evaluate Model Function
# - Used during training to get the precision, recall, and f-score for the test
##


def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    # Initialize Variables used in this function
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 in calculation
    true_negatives = 0
    false_negatives = 1e-8
    fScoreArray = [[0, 0, 0, 0, 0, 0],  # Columns are true labels and rows are predicted labels
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]]

    # Loop through each review in the batch that was passed in
    for i, review in enumerate(textcat.pipe(reviews)):
        # Get the actual label for the review
        true_label = labels[i]["cats"]

        # Loop through the predictions in the model and create the confusion matrix for the multi-class metric system
        for predicted_label, score in review.cats.items():
            if true_label['one']:
                if predicted_label == 'one':
                    fScoreArray[1][1] += 1
                elif predicted_label == 'two':
                    fScoreArray[2][1] += 1
                elif predicted_label == 'three':
                    fScoreArray[3][1] += 1
                elif predicted_label == 'four':
                    fScoreArray[4][1] += 1
                elif predicted_label == 'five':
                    fScoreArray[5][1] += 1
            elif true_label['two']:
                if predicted_label == 'one':
                    fScoreArray[1][2] += 1
                elif predicted_label == 'two':
                    fScoreArray[2][2] += 1
                elif predicted_label == 'three':
                    fScoreArray[3][2] += 1
                elif predicted_label == 'four':
                    fScoreArray[4][2] += 1
                elif predicted_label == 'five':
                    fScoreArray[5][2] += 1
            elif true_label['three']:
                if predicted_label == 'one':
                    fScoreArray[1][3] += 1
                elif predicted_label == 'two':
                    fScoreArray[2][3] += 1
                elif predicted_label == 'three':
                    fScoreArray[3][3] += 1
                elif predicted_label == 'four':
                    fScoreArray[4][3] += 1
                elif predicted_label == 'five':
                    fScoreArray[5][3] += 1
            elif true_label['four']:
                if predicted_label == 'one':
                    fScoreArray[1][4] += 1
                elif predicted_label == 'two':
                    fScoreArray[2][4] += 1
                elif predicted_label == 'three':
                    fScoreArray[3][4] += 1
                elif predicted_label == 'four':
                    fScoreArray[4][4] += 1
                elif predicted_label == 'five':
                    fScoreArray[5][4] += 1
            elif true_label['five']:
                if predicted_label == 'one':
                    fScoreArray[1][5] += 1
                elif predicted_label == 'two':
                    fScoreArray[2][5] += 1
                elif predicted_label == 'three':
                    fScoreArray[3][5] += 1
                elif predicted_label == 'four':
                    fScoreArray[4][5] += 1
                elif predicted_label == 'five':
                    fScoreArray[5][5] += 1

    # Calculate 1 star rating score
    precision1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[1]
                                      [2] + fScoreArray[1][3] + fScoreArray[1][4] + fScoreArray[1][5])
    recall1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[2]
                                   [1] + fScoreArray[3][1] + fScoreArray[4][1] + fScoreArray[5][1])
    fScore1 = 2 * (precision1 * recall1) / (precision1 + recall1)

    # Calculate 2 star rating score
    precision2 = fScoreArray[2][2] / (fScoreArray[2][1] + fScoreArray[2]
                                      [2] + fScoreArray[2][3] + fScoreArray[2][4] + fScoreArray[2][5])
    recall2 = fScoreArray[2][2] / (fScoreArray[1][2] + fScoreArray[2]
                                   [2] + fScoreArray[3][2] + fScoreArray[4][2] + fScoreArray[5][2])
    fScore2 = 2 * (precision2 * recall2) / (precision2 + recall2)

    # Calculate 3 star rating score
    precision3 = fScoreArray[3][3] / (fScoreArray[3][1] + fScoreArray[3]
                                      [2] + fScoreArray[3][3] + fScoreArray[3][4] + fScoreArray[3][5])
    recall3 = fScoreArray[3][3] / (fScoreArray[1][3] + fScoreArray[2]
                                   [3] + fScoreArray[3][3] + fScoreArray[4][3] + fScoreArray[5][3])
    fScore3 = 2 * (precision3 * recall3) / (precision3 + recall3)

    # Calculate 4 star rating score
    precision4 = fScoreArray[4][4] / (fScoreArray[4][1] + fScoreArray[4]
                                      [2] + fScoreArray[4][3] + fScoreArray[4][4] + fScoreArray[4][5])
    recall4 = fScoreArray[4][4] / (fScoreArray[1][4] + fScoreArray[2]
                                   [4] + fScoreArray[3][4] + fScoreArray[4][4] + fScoreArray[5][4])
    fScore4 = 2 * (precision4 * recall4) / (precision4 + recall4)

    # Calculate 5 star rating score
    precision5 = fScoreArray[5][5] / (fScoreArray[5][1] + fScoreArray[5]
                                      [2] + fScoreArray[5][3] + fScoreArray[5][4] + fScoreArray[5][5])
    recall5 = fScoreArray[5][5] / (fScoreArray[1][5] + fScoreArray[2]
                                   [5] + fScoreArray[3][5] + fScoreArray[4][5] + fScoreArray[5][5])
    fScore5 = 2 * (precision5 * recall5) / (precision5 + recall5)

    # Average the scores together
    macroPrecision = (precision1 + precision2 +
                      precision3 + precision4 + precision5) / 5
    macroRecall = (recall1 + recall2 + recall3 + recall4 + recall5) / 5
    macroFScore = (fScore1 + fScore2 + fScore3 + fScore4 + fScore5) / 5

    # Print individual f-scores if need to
    """
    print("\t1: prec: " + str(precision1) + " recall: " + str(recall1) + " f-score: " + str(fScore1))
    print("\t2: prec: " + str(precision2) + " recall: " + str(recall2) + " f-score: " + str(fScore2))
    print("\t3: prec: " + str(precision3) + " recall: " + str(recall3) + " f-score: " + str(fScore3))
    print("\t4: prec: " + str(precision4) + " recall: " + str(recall4) + " f-score: " + str(fScore4))
    print("\t5: prec: " + str(precision5) + " recall: " + str(recall5) + " f-score: " + str(fScore5))
    """

    # Return the average precision, recall, and f-score
    return {"precision": macroPrecision, "recall": macroRecall, "f-score": macroFScore}

##
# Test Model Function
# - This function is called when testing the batches, loads in data and predicts the value of each review followed
# by saving the results to a .csv file for further investigation
##


def test_model():
    #  Load the particular model you want to test with
    modelName = "ManuallyLabelledModel_2"  # MODEL NAME ##
    loaded_model = spacy.load(modelName)
    print("Using model: " + modelName)

    # Load spacy library for removing stopwords
    nlp = spacy.load("en_core_web_sm")

    # Initialize variables for the for-loop
    reviewIndex = 0
    reviewTextList = []
    ratingList = []
    ogRatingList = []
    scoreList = []
    fScoreArray = [[0, 0, 0, 0, 0, 0],  # Columns are true label and rows are predicted labels
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]]

    # Individual arrays to store the information to graph later on
    fScoreArrayValues1 = []
    precisionValues1 = []
    recallValues1 = []

    fScoreArrayValues2 = []
    precisionValues2 = []
    recallValues2 = []

    fScoreArrayValues3 = []
    precisionValues3 = []
    recallValues3 = []

    fScoreArrayValues4 = []
    precisionValues4 = []
    recallValues4 = []

    fScoreArrayValues5 = []
    precisionValues5 = []
    recallValues5 = []

    allReviews = []
    entry = []

    # Designate the review dataset you are using to test
    fileName = 'manuallyLabelledReviews.json'  # FILE NAME ##
    print("Loading reviews from: " + fileName)

    # Open and remove the special formatting of the .json file
    with open(fileName) as data_file:
        reviews = json.loads("[" +
                             data_file.read().replace("}\n{", "},\n{") +
                             "]")
    # Go through each review and append the review and rating to an index in the allReviews array for testing
    for review in reviews:
        entry.append(review.pop('review'))
        entry.append(review.pop('rating'))
        allReviews.append(entry)
        entry = []

    # Shuffle the reviews
    random.shuffle(allReviews)
    print("Loaded reviews and Shuffled.")
    print("\nStarting Metric Phase...")

    # Go through each review, pre process and predict rating then add to the confusion matrix for later computations
    for review in allReviews:
        # Remove Stopwords from the review
        inputData = ' '.join(
            [token.text_with_ws for token in nlp(review[0]) if not token.is_stop])

        # This is to be used when testing with stopwords
        #inputData = review[0]

        # Generate the predicting rating
        parsed_text = loaded_model(inputData)

        # Determine prediction
        if parsed_text.cats["one"] > parsed_text.cats["two"] and parsed_text.cats["one"] > parsed_text.cats["three"] and parsed_text.cats["one"] > parsed_text.cats["four"] and parsed_text.cats["one"] > parsed_text.cats["five"]:
            prediction = 1
            score = parsed_text.cats["one"]
        elif parsed_text.cats["two"] > parsed_text.cats["one"] and parsed_text.cats["two"] > parsed_text.cats["three"] and parsed_text.cats["two"] > parsed_text.cats["four"] and parsed_text.cats["two"] > parsed_text.cats["five"]:
            prediction = 2
            score = parsed_text.cats["two"]
        elif parsed_text.cats["three"] > parsed_text.cats["one"] and parsed_text.cats["three"] > parsed_text.cats["two"] and parsed_text.cats["three"] > parsed_text.cats["four"] and parsed_text.cats["three"] > parsed_text.cats["five"]:
            prediction = 3
            score = parsed_text.cats["three"]
        elif parsed_text.cats["four"] > parsed_text.cats["one"] and parsed_text.cats["four"] > parsed_text.cats["two"] and parsed_text.cats["four"] > parsed_text.cats["three"] and parsed_text.cats["four"] > parsed_text.cats["five"]:
            prediction = 4
            score = parsed_text.cats["four"]
        else:
            prediction = 5
            score = parsed_text.cats["five"]

        # Add items to lists for saving
        reviewTextList.append(review[0])
        ratingList.append(prediction)
        ogRatingList.append(review[1])
        scoreList.append(score)

        # Get TP, FP, TN, or FN
        fScoreArray[prediction][review[1]] += 1

        # Compute precision, recall, and f-score for each rating
        precision1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[1]
                                          [2] + fScoreArray[1][3] + fScoreArray[1][4] + fScoreArray[1][5] + 1e-8)
        recall1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[2][1] +
                                       fScoreArray[3][1] + fScoreArray[4][1] + fScoreArray[5][1] + 1e-8)
        fScore1 = 2 * (precision1 * recall1) / (precision1 + recall1 + 1e-8)
        precisionValues1.append(precision1)
        recallValues1.append(recall1)
        fScoreArrayValues1.append(fScore1)

        precision2 = fScoreArray[2][2] / (fScoreArray[2][1] + fScoreArray[2]
                                          [2] + fScoreArray[2][3] + fScoreArray[2][4] + fScoreArray[2][5] + 1e-8)
        recall2 = fScoreArray[2][2] / (fScoreArray[1][2] + fScoreArray[2][2] +
                                       fScoreArray[3][2] + fScoreArray[4][2] + fScoreArray[5][2] + 1e-8)
        fScore2 = 2 * (precision2 * recall2) / (precision2 + recall2 + 1e-8)
        precisionValues2.append(precision2)
        recallValues2.append(recall2)
        fScoreArrayValues2.append(fScore2)

        precision3 = fScoreArray[3][3] / (fScoreArray[3][1] + fScoreArray[3]
                                          [2] + fScoreArray[3][3] + fScoreArray[3][4] + fScoreArray[3][5] + 1e-8)
        recall3 = fScoreArray[3][3] / (fScoreArray[1][3] + fScoreArray[2][3] +
                                       fScoreArray[3][3] + fScoreArray[4][3] + fScoreArray[5][3] + 1e-8)
        fScore3 = 2 * (precision3 * recall3) / (precision3 + recall3 + 1e-8)
        precisionValues3.append(precision3)
        recallValues3.append(recall3)
        fScoreArrayValues3.append(fScore3)

        precision4 = fScoreArray[4][4] / (fScoreArray[4][1] + fScoreArray[4]
                                          [2] + fScoreArray[4][3] + fScoreArray[4][4] + fScoreArray[4][5] + 1e-8)
        recall4 = fScoreArray[4][4] / (fScoreArray[1][4] + fScoreArray[2][4] +
                                       fScoreArray[3][4] + fScoreArray[4][4] + fScoreArray[5][4] + 1e-8)
        fScore4 = 2 * (precision4 * recall4) / (precision4 + recall4 + 1e-8)
        precisionValues4.append(precision4)
        recallValues4.append(recall4)
        fScoreArrayValues4.append(fScore4)

        precision5 = fScoreArray[5][5] / (fScoreArray[5][1] + fScoreArray[5]
                                          [2] + fScoreArray[5][3] + fScoreArray[5][4] + fScoreArray[5][5] + 1e-8)
        recall5 = fScoreArray[5][5] / (fScoreArray[1][5] + fScoreArray[2][5] +
                                       fScoreArray[3][5] + fScoreArray[4][5] + fScoreArray[5][5] + 1e-8)
        fScore5 = 2 * (precision5 * recall5) / (precision5 + recall5 + 1e-8)
        precisionValues5.append(precision5)
        recallValues5.append(recall5)
        fScoreArrayValues5.append(fScore5)

    # Print Confusion Matrix if needed
    total = 0
    for row in fScoreArray:
        for elem in row:
            #print(elem, end=' ')
            total += elem
        # print()

    # Print the individual precision, recall, and f-score for each rating
    print("\n      1 : " + str(precisionValues1[-1]) + "\t" + str(
        recallValues1[-1]) + "\t" + str(fScoreArrayValues1[-1]))
    print("      2 : " + str(precisionValues2[-1]) + "\t" + str(
        recallValues2[-1]) + "\t" + str(fScoreArrayValues2[-1]))
    print("      3 : " + str(precisionValues3[-1]) + "\t" + str(
        recallValues3[-1]) + "\t" + str(fScoreArrayValues3[-1]))
    print("      4 : " + str(precisionValues4[-1]) + "\t" + str(
        recallValues4[-1]) + "\t" + str(fScoreArrayValues4[-1]))
    print("      5 : " + str(precisionValues5[-1]) + "\t" + str(
        recallValues5[-1]) + "\t" + str(fScoreArrayValues5[-1]))

    # Print the final averaged precision, recall, and f-score
    precision = (precisionValues1[-1] + precisionValues2[-1] +
                 precisionValues3[-1] + precisionValues4[-1] + precisionValues5[-1]) / 5
    recall = (recallValues1[-1] + recallValues2[-1] +
              recallValues3[-1] + recallValues4[-1] + recallValues5[-1]) / 5
    fScore = (fScoreArrayValues1[-1] + fScoreArrayValues2[-1] +
              fScoreArrayValues3[-1] + fScoreArrayValues4[-1] + fScoreArrayValues5[-1]) / 5
    print("\nCombined: " + str(precision) +
          "\t" + str(recall) + "\t" + str(fScore))
    print("Accuracy: " + str((fScoreArray[1][1] + fScoreArray[2][2] +
          fScoreArray[3][3] + fScoreArray[4][4] + fScoreArray[5][5]) / total))

    # Save Review Information to .csv
    df = pd.DataFrame(data={"Review Text": reviewTextList, "Model Rating": ratingList,
                      "Reviewer Rating": ogRatingList, "Score": scoreList})
    reviewSaveFileName = "./originalReviews.csv"  # FILE NAME ##
    df.to_csv(reviewSaveFileName, sep=',', index=False)
    print("\nFile: " + reviewSaveFileName + " saved.")

    # Save metrics to .csv
    df = pd.DataFrame(data={"Precision 1": precisionValues1, "Recall 1": recallValues1, "F-Score 1": fScoreArrayValues1, "Precision 2": precisionValues2, "Recall 2": recallValues2, "F-Score 2": fScoreArrayValues2, "Precision 3": precisionValues3,
                      "Recall 3": recallValues3, "F-Score 3": fScoreArrayValues3, "Precision 4": precisionValues4, "Recall 4": recallValues4, "F-Score 4": fScoreArrayValues4, "Precision 5": precisionValues5, "Recall 5": recallValues5, "F-Score 5": fScoreArrayValues5})
    metricsSaveFileName = "./originalMetrics.csv"  # FILE NAME ##
    df.to_csv(metricsSaveFileName, sep=',', index=False)
    print("File: " + metricsSaveFileName + " saved.")
    print("Done Metric Phase.")

##
# Test Single Review Function
# - This function is called when testing a single review, loads in data and predicts the value of each review followed
# by outputting the prediction and score to the user
##


def test_model_single():
    #  Load the particular model you want to test with
    modelName = "ManuallyLabelledModel_2"  # MODEL NAME ##
    loaded_model = spacy.load(modelName)
    print("Using model: " + modelName)

    # Get review from user
    print("Enter Review Text: ")
    input_data = str(input())

    # Load spacy library for removing stopwords
    nlp = spacy.load("en_core_web_sm")
    # remove the stopwords from the review
    new_input_data = ' '.join(
        [token.text_with_ws for token in nlp(input_data) if not token.is_stop])

    # Generate prediction
    parsed_text = loaded_model(new_input_data)

    # Print the score for each rating
    """
    print()
    print("1: " + str(parsed_text.cats['one']))
    print("2: " + str(parsed_text.cats['two']))
    print("3: " + str(parsed_text.cats['three']))
    print("4: " + str(parsed_text.cats['four']))
    print("5: " + str(parsed_text.cats['five']))
    print() 
    """

    # Determine prediction to return
    if parsed_text.cats["one"] > parsed_text.cats["two"] and parsed_text.cats["one"] > parsed_text.cats["three"] and parsed_text.cats["one"] > parsed_text.cats["four"] and parsed_text.cats["one"] > parsed_text.cats["five"]:
        prediction = "1/5"
        score = parsed_text.cats["one"]
    elif parsed_text.cats["two"] > parsed_text.cats["one"] and parsed_text.cats["two"] > parsed_text.cats["three"] and parsed_text.cats["two"] > parsed_text.cats["four"] and parsed_text.cats["two"] > parsed_text.cats["five"]:
        prediction = "2/5"
        score = parsed_text.cats["two"]
    elif parsed_text.cats["three"] > parsed_text.cats["one"] and parsed_text.cats["three"] > parsed_text.cats["two"] and parsed_text.cats["three"] > parsed_text.cats["four"] and parsed_text.cats["three"] > parsed_text.cats["five"]:
        prediction = "3/5"
        score = parsed_text.cats["three"]
    elif parsed_text.cats["four"] > parsed_text.cats["one"] and parsed_text.cats["four"] > parsed_text.cats["two"] and parsed_text.cats["four"] > parsed_text.cats["three"] and parsed_text.cats["four"] > parsed_text.cats["five"]:
        prediction = "4/5"
        score = parsed_text.cats["four"]
    else:
        prediction = "5/5"
        score = parsed_text.cats["five"]

    # Show results of the prediction
    print("----\nPredicted Rating: " + str(prediction) +
          "\tScore: " + str(score) + "\n----")

##
# Load Training Data Function
# - This function is called when testing a single review, loads in data and predicts the value of each review followed
# by outputting the prediction and score to the user
##


def load_training_data(data_directory: str = "./Labelled Reviews/Personally Labelled/train", split: float = 0.8, limit: int = 0) -> tuple:
    # Display what information is being loaded in
    print("Loading training data from: " + data_directory)

    # Load the spaCy library for removing stopwords, etc.
    nlp = spacy.load("en_core_web_sm")

    # Go through the files in the directory
    reviews = []
    for label in ["one", "two", "three", "four", "five"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            # Verify the file is the right type
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
                    # Read in review, replace special characters, and remove stopwords
                    text = f.read()
                    text = text.replace("<br />", " ")
                    text = text.replace("\n", " ")
                    stopwordlessText = ' '.join(
                        [token.text_with_ws for token in nlp(text) if not token.is_stop])
                    stopwordlessText = text
                    # Essentially, here we are assigning the rating to the review for the training portion
                    if stopwordlessText.strip():
                        spacy_label = {"cats": {"one": "one" == label, "two": "two" == label,
                                                "three": "three" == label, "four": "four" == label, "five": "five" == label}}
                        reviews.append((stopwordlessText, spacy_label))
    # Shuffle reviews
    random.shuffle(reviews)

    # Only allow up the limit amount of reviews, this value comes from the main function
    if limit:
        reviews = reviews[:limit]
    # Splits the reviews loaded in half, one half for training and the other for testing, both used in the training process
    split = int(len(reviews) * split)

    # Return the training and testing set
    return reviews[:split], reviews[split:]


##
# Main Function
##
if __name__ == "__main__":
    # Get value from command line to determine what needs to be done
    if sys.argv[1] == 'train':
        train, test = load_training_data(limit=25000)
        print("Training model...")
        train_model(train, test)
    elif sys.argv[1] == 'test':
        if sys.argv[2] == 'batch':
            print("Testing model...")
            test_model()
        elif sys.argv[2] == 'single':
            test_model_single()
