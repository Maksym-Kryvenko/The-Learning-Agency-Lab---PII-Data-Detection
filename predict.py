from tqdm.notebook import tqdm
from main import ModelConfiguration
import numpy as np
import pandas as pd


# Your prediction code here
def make_predictions(model, processed_test_data):
    # ...
    print("PREDICTING: Preparing data ...")
    words, docs, test_token_ids, test_ds = processed_test_data

    # INFERENCE
    print("PREDICTING: Predicting the results ...")
    test_preds = model.predict(test_ds, verbose=1)

    # Convert probabilities to class labels via max confidence
    print("PREDICTING: Converting probabilities into classes ...")
    test_preds = np.argmax(test_preds, axis=-1)

    # POST PROCESSING
    document_list = []
    token_id_list = []
    label_id_list = []
    token_list = []

    print("PREDICTING: Post processing data ... ")
    for doc, token_ids, preds, tokens in tqdm(
        zip(docs, test_token_ids, test_preds, words), total=len(words)
    ):
        # Create mask for filtering
        mask1 = np.concatenate(([True], token_ids[1:] != token_ids[:-1])) # ignore non-start tokens of a word
        mask2 = (preds != 9)                                              # ignore `O` (BIO format) label -> 9 (integer format) label
        mask3 = (token_ids != -1)                                         # ignore [CLS], [SEP], and [PAD] tokens
        mask = (mask1 & mask2 & mask3)                                    # merge filters
        
        # Apply filter
        token_ids = token_ids[mask]
        preds = preds[mask]

        # Store prediction if number of tokens is not zero
        if len(token_ids):
            token_list.extend(tokens[token_ids])
            document_list.extend([doc] * len(token_ids))
            token_id_list.extend(token_ids)
            label_id_list.extend(preds)

    # SUBMISSION
    pred_df = pd.DataFrame(
        {
            "document": document_list,
            "token": token_id_list,
            "label_id": label_id_list,
            "token_string": token_list,
        }
    )
    pred_df = pred_df.rename_axis("row_id").reset_index() # add `row_id` column
    pred_df["label"] = pred_df.label_id.map(ModelConfiguration.id2label) # map integer label to BIO format label

    # SAVE FILE
    print("PREDICTING: Saving file with PII ...")
    sub_df = pred_df.drop(columns=["token_string", "label_id"]) # remove extra columns
    sub_df.to_csv("submission.csv", index=False)

    # **** WITH SPACY SAVE TEXT WITHOUT THE PII WORDS **** #
    print("PREDICTING: Saving processed data ...")
    # TBD

    print("PREDICTING: Done!")
