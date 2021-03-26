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

TEST_REVIEW2 = """
The stated premise of the book, that strangers can improve your insight, innovation, and success, is true and is a relevant topic.  Gregerman presents a genial book that I wanted to like. OK, I did like it, but was still left disappointed because it didn't really deliver what it promised.  This isn't really a book about strangers, or being open to the influence of strangers, it's about openmindedness in general.  That's a fine topic too, but one that I have already read and thought extensively about.  That is why I was excited to find a book that would delve deep into one aspect of that -- openmindedness toward strangers.  Alas, this did not deliver.  IF you are looking for a fun and light, but still useful, book on openmindedness, in general, you should be very pleased.  This book offers a shallow look at many dimensions of openmindedness.\n\nEarly on, Gregerman writes, \"It may seem counterintuitive that strangers are to be embraced rather than ignored or avoided, but they are a necessity -- precisely because of their differences and what they know that we don't know; their objectivity and ability to be open and honest with us about the things that really matter; and their capacity to challenge us to think very differently about ourselves, the problems we face, and the nature of what is possible.  Finding and engaging the right strangers has the power to make all of us more complete, compelling, innovative, and successful.\"  This is exactly what I wanted, but Gregerman doesn't go into details on this.  He basically just tells us to talk with people.  Very general.  Indeed, the first chapter, Necessity, would have made a great article or Kindle Short, as that is where the heart of the idea is.  It's a great chapter in an otherwise average book.\n\nIn chapter 4, Innovation, Gregerman tells us to ask ourselves these 8 questions:  What's our best thinking to date?  What's the best thinking in our industry?  What's the best thinking in other industries?  What's the best thinking from popular culture?  What's the best thinking in other cultures?  What's the best insight from nature?  What's the best insight from science?  What are the best possibilities from science fiction?  OK, good questions, and Gregerman gives each a paragraph telling us why it is a good question, but does not give guidance on how to actually answer the questions.\n\nGregerman really stretches the definition of stranger.  At one point he uses fish as a \"stranger\" and discusses how the schooling behavior of fish led to innovations in accident avoiding cars.  Really, fish as strangers?  Isn't this really about openmindedness in general, and not openmindedness to strangers specifically?  Gregerman explicitly sidesteps the issue of strangers by stating that his approach is \"driven by the reality that in many ways almost all of us are strangers.\"  So, are we talking about openness to strangers, or about openness to people in general?  Why did the title of this book not refer to openmindedness rather than strangers?\n\nSome other books also offer insight into what strangers have to offer.  Check out Give And Take, by Adam Grant, which has more to offer than this book does, even though the premise is a bit different.  The Geography of Happiness may also be worth a look, though it is primarily about personal benefit, not business; but of course, these are entwined.  Another book, that Gregerman refers to is Carol Dweck's Mindset: The New Psychology of Success, which is specifically about openmindedness and closemindedness.\n\nGregerman seems like someone I would love to meet in person and talk with.  Really wish that he had examined the stated topic in more detail so that I could leave a more enthusiastic review.  Gergerman's last paragraph is a good summation of his book, so I present it here.  If this is what you actually want to read about, rather than an in-depth look at strangers, then you should enjoy this book:  \"Use these resources as a starting point for stretching your thinking or as a call to action to look in new places for ideas that could make a real difference to you and your organization.  Then commit to the regular habit of picking up a book or ebook, reading a magazine or blog, or watching a movie or TV show that takes you off of your well-beaten path -- not because everyone else around you is doing so, but because it might take you to a new and remarkable place where the possibility of a breakthrough awaits.\"
"""

TEST_REVIEW3 = """
First I'll say that I am very happy these boxes exist. I love using them. That being said this box is no better than any other. Same old cheap crap in a different form. I have no idea why these manufacturers cannot add gaskets or silicone rings to these things! I'd be willing to pay 10 dollars more if there was a gasket on the outside of the inlet and outlet, around the closer (door) and on the inside of the little lid. These things leak air like crazy. I use thin foil heat tape and wrap every possible area that might leak. The zip ties do nothing, there is nowhere for them to grip to. There is no lip to keep it from sliding off. Taped that too. The door that swings with the adjuster leaks air too. It's fine in the winter. Horrible in the summer. I also had to tape over the screen and around the lid to seal it. I'll just remove the tape for winter.\n\nI spent 40 minutes taping and testing. It's on and working like it should but one should not have to put so much time and effort into such a simple contraption. I do recommend these boxes but you have to have realistic expectations. It's not going to work without lots of fuss.
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

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
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
                    f"{loss['textcat']}\t\t{evaluation_results['precision']}"
                    f"\t\t{evaluation_results['recall']}"
                    f"\t\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")


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


def test_model(input_data: str = TEST_REVIEW3):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
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


def load_training_data(
    data_directory: str = "aclImdb/train", split: float = 0.8, limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["pos", "neg"]:
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
        train, test = load_training_data(limit=2500)
        print("Training model")
        train_model(train, test)
        
        #df = pd.DataFrame(eval_list)
        #pd.DataFrame.plot(df)
        print("Testing model")
        test_model()
    elif sys.argv[1] == 'test':
        #df = pd.DataFrame(eval_list)
        #pd.DataFrame.plot(df)
        print("Testing model")
        test_model()