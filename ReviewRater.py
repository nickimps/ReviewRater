#
# ESOF 2918 Technical Project
#
# Review Rater Project - analyzes sentiment in reviews
#
# Nicholas Imperius, Kristopher Pouling, Jimmy Tsang
#

import os
import sys
import random
import spacy
from spacy.util import minibatch, compounding
#import pandas as pd


TEST_REVIEW = """
Transcendently beautiful in moments outside the office, it seems almost
sitcom-like in those scenes. When Toni Colette walks out and ponders
life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide
whether it's slapstick, farce, magical realism, or drama, but the best of it
doesn't matter. (The worst is sort of tedious - like Office Space with less
humor.)
"""

TEST_REVIEW_POS = """
Work perfectly in my Frigidaire fridge/ice maker. My original filter was so clogged with 
iron/calcium from our hard water and iron pipes that my ice cubes were hollow and the water 
dispenser was pretty much unusable. I don't know why it didn't occur to me to check the filter, 
but I'm glad I did. I have ice again! Huzzah!
"""

TEST_REVIEW_NEG = """
Doesn't work, no water flow. It's obvious from comparing the top 
of the OEM and this filter that it's not the same.
"""

TEST_REVIEW_NEUT = """
works great but makes small ice.
"""


eval_list = []

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

    textcat.add_label("pos")
    textcat.add_label("neg")
    textcat.add_label("neut")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training...")
        print("Loss\t\tPrecision\t\tRecall\t\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
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
        nlp.to_disk("model_artifacts_2") ################ _2 is the neww one


def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]["cats"]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels, you can get all
            # the info you need with just the pos label
            if predicted_label == "neg":
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def test_model(input_data: str = TEST_REVIEW_NEG):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts_2") ################# _2 is the neww one
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    print()
    print("pos: " + str(parsed_text.cats['pos']) + "\tneg: " + str(parsed_text.cats['neg']) + "\tneut: " + str(parsed_text.cats['neut']))
    print()
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )

#
# - if time we can create our own training folder of reviews and sort them with 
#   positive or negative reviews to train the system
#
def load_training_data(
    data_directory: str = "labelledReviews/train", split: float = 0.8, limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["pos", "neg", "neut"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label,
                                "neut": "neut" == label,
                            }
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        print("Loading training data...")
        train, test = load_training_data(limit=2199)
        print("Training model...")
        train_model(train, test)
        
    if sys.argv[1] == 'test' or sys.argv[1] == 'train':
        #df = pd.DataFrame(eval_list)
        #pd.DataFrame.plot(df)
        print("Testing model...")
        test_model()