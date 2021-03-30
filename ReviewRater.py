#
# ESOF 2918 Technical Project
#
# Review Rater Project - analyzes sentiment in reviews
#
# Nicholas Imperius, Kristopher Poulin, Jimmy Tsang
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

TEST_REVIEW_NEUT_2 = """
Quality product that upon first glance appears to be a better 
made replacement than the original.
"""

TEST_REVIEW_ONE = """
We just moved in a house with a 30" Kenmore elite gas cooktop and there are several negative 
things about it.  The worst thing is that both the front plate and the first knob that controls 
the temp get hot enough to literally burn you if touched.  We have been told that is a safety issue, 
either gas is leaking out or there is an electrical short.  The second thing is that one of the front 
burners went always turn on when you turn the knob.  Some times it only clicks and then you turn it off 
and on again and a big flame shoots out.
Sears will not do anything about it, they only want to sell you a new one.  NEVER BUY A KENMORE!
"""

TEST_REVIEW_TWO = """
Loose filtration media in the filter, resulted in having to replace the water inlet valve in my refrigerator.

Upon replacing the filter, my ice maker fill valve would not fully shut off.  Water continued to drip into the 
ice tray, overflowing into the freezer.  This resulted in having to replace the water inlet valve.

Now that the valve has been replaced, the filter does a great job at making the water taste good.  But any 
savings I had by buying this filter, instead of a Maytag OEM filter, was lost three times over by the expense 
of having to replace the valve.

If you decide to buy this filter, be very thorough in rinsing it out before install, taking care to assure 
that no loose debris comes out.

Personally, next time I will buy the Maytag filter.
"""

TEST_REVIEW_THREE = """
I purchased this fridge from another vendor and I was happy with it except it died after about 2 years of use. 
I looked into getting it repaired, but the person I spoke to said it would be more cost effective to replace it 
due to the labor charges. I saw on the Target website that another person had the same experience, so it seems 
the longevity of this product is in question.
"""

TEST_REVIEW_FOUR = """
I love this little guy. It washes clothes well. They come out practical dry out of the spinner. It amazed me.
"""

TEST_REVIEW_FIVE = """
Decided to fix my dryer on my own instead of shelling out over $100 for a repair. With this part and the 
Thermostat Fuse I did the fix for $20. It has been several months and the dryer is still working like new.
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
        print("Loss\t\tPrecision\t\tRecall\t\tF-score")
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
        nlp.to_disk("model_artifacts_3") ################


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
            #print("  pl: " + str(predicted_label))
            #print("  sc: " + str(score))
            #print("  tl: " + str(true_label))
            if true_label['one'] and predicted_label == 'one':
                if score >= 0.5:
                    true_positives += 1
                else:
                    false_positives += 1
            elif true_label['one'] and predicted_label != 'one':
                if score >= 0.5:
                    true_negatives += 1
                else:
                    false_negatives += 1
            elif true_label['two'] and predicted_label == 'two':
                if score >= 0.5:
                    true_positives += 1
                else:
                    false_positives += 1
            elif true_label['two'] and predicted_label != 'two':
                if score >= 0.5:
                    true_negatives += 1
                else:
                    false_negatives += 1
            elif true_label['three'] and predicted_label == 'three':
                if score >= 0.5:
                    true_positives += 1
                else:
                    false_positives += 1
            elif true_label['three'] and predicted_label != 'three':
                if score >= 0.5:
                    true_negatives += 1
                else:
                    false_negatives += 1
            elif true_label['four'] and predicted_label == 'four':
                if score >= 0.5:
                    true_positives += 1
                else:
                    false_positives += 1
            elif true_label['four'] and predicted_label != 'four':
                if score >= 0.5:
                    true_negatives += 1
                else:
                    false_negatives += 1
            elif true_label['five'] and predicted_label == 'five':
                if score >= 0.5:
                    true_positives += 1
                else:
                    false_positives += 1
            elif true_label['five'] and predicted_label != 'five':
                if score >= 0.5:
                    true_negatives += 1
                else:
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
    loaded_model = spacy.load("model_artifacts_3") # 1 - pos, neg , 2 - pos, neut, neg , 3 - one,..., five 
    
    # removes stopwords from input data
    nlp = spacy.load("en_core_web_sm")
    new_input_data = ' '.join([token.text_with_ws for token in nlp(input_data) if not token.is_stop])
    
    print("Text w/o stopwords: " + new_input_data)
    
    # Generate prediction
    parsed_text = loaded_model(new_input_data)
    # Determine prediction to return
    print()
    print("one: " + str(parsed_text.cats['one']))
    print("two: " + str(parsed_text.cats['two']))
    print("three: " + str(parsed_text.cats['three']))
    print("four: " + str(parsed_text.cats['four']))
    print("five: " + str(parsed_text.cats['five']))
    print() 
    
    if parsed_text.cats["one"] > parsed_text.cats["two"] and parsed_text.cats["one"] > parsed_text.cats["three"] and parsed_text.cats["one"] > parsed_text.cats["four"] and parsed_text.cats["one"] > parsed_text.cats["five"]:
        prediction = "1 Star"
        score = parsed_text.cats["one"]
    elif parsed_text.cats["two"] > parsed_text.cats["one"] and parsed_text.cats["two"] > parsed_text.cats["three"] and parsed_text.cats["two"] > parsed_text.cats["four"] and parsed_text.cats["two"] > parsed_text.cats["five"]:
        prediction = "2 Star"
        score = parsed_text.cats["two"]
    elif parsed_text.cats["three"] > parsed_text.cats["one"] and parsed_text.cats["three"] > parsed_text.cats["two"] and parsed_text.cats["three"] > parsed_text.cats["four"] and parsed_text.cats["three"] > parsed_text.cats["five"]:
        prediction = "3 Star"
        score = parsed_text.cats["three"]
    elif parsed_text.cats["four"] > parsed_text.cats["one"] and parsed_text.cats["four"] > parsed_text.cats["two"] and parsed_text.cats["four"] > parsed_text.cats["three"] and parsed_text.cats["four"] > parsed_text.cats["five"]:
        prediction = "4 Star"
        score = parsed_text.cats["four"]
    else:
        prediction = "5 Star"
        score = parsed_text.cats["five"] 

    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )

#
# - if time we can create our own training folder of reviews and sort them with 
#   positive or negative reviews to train the system
#
def load_training_data(
    data_directory: str = "dataset/train", split: float = 0.8, limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["one", "two", "three", "four", "five"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "one": "one" == label,
                                "two": "two" == label,
                                "three": "three" == label,
                                "four": "four" == label,
                                "five": "five" == label,
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
        train, test = load_training_data(limit=2000)
        print("Training model...")
        train_model(train, test)
        
    if sys.argv[1] == 'test' or sys.argv[1] == 'train':
        #df = pd.DataFrame(eval_list)
        #pd.DataFrame.plot(df)
        print("Testing model...")
        test_model()