#
# ESOF 2918 Technical Project
#
# Review Rater Project - analyzes sentiment in reviews
#
# Nicholas Imperius, Kristopher Poulin, Jimmy Tsang
#

import os, json
import sys
import random
import spacy
from spacy.util import minibatch, compounding
import pandas as pd

def train_model(
    training_data: list, test_data: list, iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("one")
    textcat.add_label("two")
    textcat.add_label("three")
    textcat.add_label("four")
    textcat.add_label("five")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training...")
        print("Loss\t\t\tPrecision\t\tRecall\t\t\tF-score")
        batch_sizes = compounding( 4.0, 32.0, 1.001 )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data,
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("ReviewerLabelledModel_3") # MODEL NAME


def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0
    true_negatives = 0
    false_negatives = 1e-8
    fScoreArray = [[0, 0, 0, 0, 0, 0], # Columns are true label and rows are predicted labels
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]]
                   
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]["cats"]
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
    
    #Calculate each rating score
    precision1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[1][2] + fScoreArray[1][3] + fScoreArray[1][4] + fScoreArray[1][5])
    recall1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[2][1] + fScoreArray[3][1] + fScoreArray[4][1] + fScoreArray[5][1])
    fScore1 = 2 * (precision1 * recall1) / (precision1 + recall1)
    #print("\t1: prec: " + str(precision1) + " recall: " + str(recall1) + " f-score: " + str(fScore1))
    
    precision2 = fScoreArray[2][2] / (fScoreArray[2][1] + fScoreArray[2][2] + fScoreArray[2][3] + fScoreArray[2][4] + fScoreArray[2][5])
    recall2 = fScoreArray[2][2] / (fScoreArray[1][2] + fScoreArray[2][2] + fScoreArray[3][2] + fScoreArray[4][2] + fScoreArray[5][2])
    fScore2 = 2 * (precision2 * recall2) / (precision2 + recall2)
    #print("\t2: prec: " + str(precision2) + " recall: " + str(recall2) + " f-score: " + str(fScore2))
    
    precision3 = fScoreArray[3][3] / (fScoreArray[3][1] + fScoreArray[3][2] + fScoreArray[3][3] + fScoreArray[3][4] + fScoreArray[3][5])
    recall3 = fScoreArray[3][3] / (fScoreArray[1][3] + fScoreArray[2][3] + fScoreArray[3][3] + fScoreArray[4][3] + fScoreArray[5][3])
    fScore3 = 2 * (precision3 * recall3) / (precision3 + recall3)
    #print("\t3: prec: " + str(precision3) + " recall: " + str(recall3) + " f-score: " + str(fScore3))
    
    precision4 = fScoreArray[4][4] / (fScoreArray[4][1] + fScoreArray[4][2] + fScoreArray[4][3] + fScoreArray[4][4] + fScoreArray[4][5])
    recall4 = fScoreArray[4][4] / (fScoreArray[1][4] + fScoreArray[2][4] + fScoreArray[3][4] + fScoreArray[4][4] + fScoreArray[5][4])
    fScore4 = 2 * (precision4 * recall4) / (precision4 + recall4)
    #print("\t4: prec: " + str(precision4) + " recall: " + str(recall4) + " f-score: " + str(fScore4))
    
    precision5 = fScoreArray[5][5] / (fScoreArray[5][1] + fScoreArray[5][2] + fScoreArray[5][3] + fScoreArray[5][4] + fScoreArray[5][5])
    recall5 = fScoreArray[5][5] / (fScoreArray[1][5] + fScoreArray[2][5] + fScoreArray[3][5] + fScoreArray[4][5] + fScoreArray[5][5])
    fScore5 = 2 * (precision5 * recall5) / (precision5 + recall5)
    #print("\t5: prec: " + str(precision5) + " recall: " + str(recall5) + " f-score: " + str(fScore5))
    
    #Average the scores together
    macroPrecision = (precision1 + precision2 + precision3 + precision4 + precision5) / 5
    macroRecall = (recall1 + recall2 + recall3 + recall4 + recall5) / 5
    macroFScore = (fScore1 + fScore2 + fScore3 + fScore4 + fScore5) / 5

    return {"precision": macroPrecision, "recall": macroRecall, "f-score": macroFScore}

def test_model():
    #  Load saved trained model
    loaded_model = spacy.load("ManuallyLabelledModel") # MODEL NAME
    
    nlp = spacy.load("en_core_web_sm")

    reviewIndex = 0
    reviewTextList = []
    ratingList = []
    ogRatingList = []
    scoreList = []
    fScoreArray = [[0, 0, 0, 0, 0, 0], # Columns are true label and rows are predicted labels
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]]
                   
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
    with open('originallyLabelledReviews.json') as data_file:
        reviews = json.loads("[" + 
            data_file.read().replace("}\n{", "},\n{") + 
        "]")
    for review in reviews:
        entry.append(review.pop('review'))
        entry.append(review.pop('rating'))
        allReviews.append(entry)
        entry = []
    
    random.shuffle(allReviews)
    print("Loaded Testing Reviews and Shuffled.")
    print("Starting Metric Phase...")
    
    for review in allReviews:
        #Remove Stopwords
        #inputData = ' '.join([token.text_with_ws for token in nlp(review[0]) if not token.is_stop])
        inputData = review[0]
        
        # Generate prediction
        parsed_text = loaded_model(inputData)
        
        # Determine prediction to return
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
    
        # Get TP, FP, TN, FN
        fScoreArray[prediction][review[1]] += 1
        
        # Compute precision, recall, and f-score for each rating
        precision1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[1][2] + fScoreArray[1][3] + fScoreArray[1][4] + fScoreArray[1][5] + 1e-8)
        recall1 = fScoreArray[1][1] / (fScoreArray[1][1] + fScoreArray[2][1] + fScoreArray[3][1] + fScoreArray[4][1] + fScoreArray[5][1] + 1e-8)
        fScore1 = 2 * (precision1 * recall1) / (precision1 + recall1 + 1e-8)
        precisionValues1.append(precision1)
        recallValues1.append(recall1)
        fScoreArrayValues1.append(fScore1)
        
        precision2 = fScoreArray[2][2] / (fScoreArray[2][1] + fScoreArray[2][2] + fScoreArray[2][3] + fScoreArray[2][4] + fScoreArray[2][5] + 1e-8)
        recall2 = fScoreArray[2][2] / (fScoreArray[1][2] + fScoreArray[2][2] + fScoreArray[3][2] + fScoreArray[4][2] + fScoreArray[5][2] + 1e-8)
        fScore2 = 2 * (precision2 * recall2) / (precision2 + recall2 + 1e-8)
        precisionValues2.append(precision2)
        recallValues2.append(recall2)
        fScoreArrayValues2.append(fScore2)
        
        precision3 = fScoreArray[3][3] / (fScoreArray[3][1] + fScoreArray[3][2] + fScoreArray[3][3] + fScoreArray[3][4] + fScoreArray[3][5] + 1e-8)
        recall3 = fScoreArray[3][3] / (fScoreArray[1][3] + fScoreArray[2][3] + fScoreArray[3][3] + fScoreArray[4][3] + fScoreArray[5][3] + 1e-8)
        fScore3 = 2 * (precision3 * recall3) / (precision3 + recall3 + 1e-8)
        precisionValues3.append(precision3)
        recallValues3.append(recall3)
        fScoreArrayValues3.append(fScore3)
        
        precision4 = fScoreArray[4][4] / (fScoreArray[4][1] + fScoreArray[4][2] + fScoreArray[4][3] + fScoreArray[4][4] + fScoreArray[4][5] + 1e-8)
        recall4 = fScoreArray[4][4] / (fScoreArray[1][4] + fScoreArray[2][4] + fScoreArray[3][4] + fScoreArray[4][4] + fScoreArray[5][4] + 1e-8)
        fScore4 = 2 * (precision4 * recall4) / (precision4 + recall4 + 1e-8)
        precisionValues4.append(precision4)
        recallValues4.append(recall4)
        fScoreArrayValues4.append(fScore4)
        
        precision5 = fScoreArray[5][5] / (fScoreArray[5][1] + fScoreArray[5][2] + fScoreArray[5][3] + fScoreArray[5][4] + fScoreArray[5][5] + 1e-8)
        recall5 = fScoreArray[5][5] / (fScoreArray[1][5] + fScoreArray[2][5] + fScoreArray[3][5] + fScoreArray[4][5] + fScoreArray[5][5] + 1e-8)
        fScore5 = 2 * (precision5 * recall5) / (precision5 + recall5 + 1e-8)
        precisionValues5.append(precision5)
        recallValues5.append(recall5)
        fScoreArrayValues5.append(fScore5)
        
    # Print for user
    total = 0
    for row in fScoreArray:
        for elem in row:
            print(elem, end=' ')
            total += elem
        print()
    
    print("\n      1 : " + str(precisionValues1[-1]) + "\t" + str(recallValues1[-1]) + "\t" + str(fScoreArrayValues1[-1]))
    print("      2 : " + str(precisionValues2[-1]) + "\t" + str(recallValues2[-1]) + "\t" + str(fScoreArrayValues2[-1]))
    print("      3 : " + str(precisionValues3[-1]) + "\t" + str(recallValues3[-1]) + "\t" + str(fScoreArrayValues3[-1]))
    print("      4 : " + str(precisionValues4[-1]) + "\t" + str(recallValues4[-1]) + "\t" + str(fScoreArrayValues4[-1]))
    print("      5 : " + str(precisionValues5[-1]) + "\t" + str(recallValues5[-1]) + "\t" + str(fScoreArrayValues5[-1]))

    precision = (precisionValues1[-1] + precisionValues2[-1] + precisionValues3[-1] + precisionValues4[-1] + precisionValues5[-1]) / 5
    recall = (recallValues1[-1] + recallValues2[-1] + recallValues3[-1] + recallValues4[-1] + recallValues5[-1]) / 5
    fScore = (fScoreArrayValues1[-1] + fScoreArrayValues2[-1] + fScoreArrayValues3[-1] + fScoreArrayValues4[-1] + fScoreArrayValues5[-1]) / 5
    print("Combined: " + str(precision) + "\t" + str(recall) + "\t" + str(fScore))
    print("Accuracy: " + str( (fScoreArray[1][1] + fScoreArray[2][2] + fScoreArray[3][3] + fScoreArray[4][4] + fScoreArray[5][5]) / total))
    
    # Save Review Information to .csv
    df = pd.DataFrame(data={"Review Text": reviewTextList, "Model Rating": ratingList, "Reviewer Rating": ogRatingList, "Score": scoreList})
    df.to_csv("./originalReviews.csv", sep=',', index=False)
    
    # Save metrics to .csv
    df = pd.DataFrame(data={"Precision 1": precisionValues1, "Recall 1": recallValues1, "F-Score 1": fScoreArrayValues1, "Precision 2": precisionValues2, "Recall 2": recallValues2, "F-Score 2": fScoreArrayValues2, "Precision 3": precisionValues3, "Recall 3": recallValues3, "F-Score 3": fScoreArrayValues3, "Precision 4": precisionValues4, "Recall 4": recallValues4, "F-Score 4": fScoreArrayValues4, "Precision 5": precisionValues5, "Recall 5": recallValues5, "F-Score 5": fScoreArrayValues5})
    df.to_csv("./originalMetrics.csv", sep = ',', index = False)
    print("Done Metric Phase.")


def test_model_single():
    #  Load saved trained model
    loaded_model = spacy.load("ManuallyLabelledModel") # MODEL NAME
    
    # Get review from user
    print("Enter Review Text: ")
    input_data = str(input())
    
    # removes stopwords from input data
    nlp = spacy.load("en_core_web_sm")
    new_input_data = ' '.join([token.text_with_ws for token in nlp(input_data) if not token.is_stop])
    
    print(new_input_data)
    
    # Generate prediction
    parsed_text = loaded_model(input_data)
    
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
    print(
        f"----\nPredicted Rating: {prediction}"
        f"\tScore: {score}\n----"
    )


def load_training_data( 
    data_directory: str = "C:\\Users\\nimpe\\Google Drive\\LU - Courses\\ESOF 2918\\Reviewer Labelled\\train", 
    split: float = 0.8, limit: int = 0 
) -> tuple:
    nlp = spacy.load("en_core_web_sm")
    # Load from files
    reviews = []
    for label in ["one", "two", "three", "four", "five"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
                    text = f.read()
                    text = text.replace("<br />", " ")
                    text = text.replace("\n", " ")
                    #stopwordlessText = ' '.join([token.text_with_ws for token in nlp(text) if not token.is_stop])
                    stopwordlessText = text
                    if stopwordlessText.strip():
                        spacy_label = {
                            "cats": {
                                "one": "one" == label,
                                "two": "two" == label,
                                "three": "three" == label,
                                "four": "four" == label,
                                "five": "five" == label,
                            }
                        }
                        reviews.append((stopwordlessText, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        print("Loading training data...")
        train, test = load_training_data(limit=2000)
        print("Training model...")
        train_model(train, test)
    elif sys.argv[1] == 'test':
        if sys.argv[2] == 'batch':
            print("Testing model...")
            test_model()
        elif sys.argv[2] == 'single':
            test_model_single()