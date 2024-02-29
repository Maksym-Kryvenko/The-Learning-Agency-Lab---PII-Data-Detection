from tqdm.notebook import tqdm
from config import ModelConfiguration, configure_logging
import numpy as np
import pandas as pd
import spacy

logging = configure_logging()  # Configure logging from config.py

def remove_pii(text, pred_df):
    # Load spaCy model with NER component
    nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    # Identify named entities and replace them
    replaced_text = " ".join([token.text for token in doc if token.text not in pred_df.words.to_list()])
    return replaced_text

# saving file with PII items
def save_predictions(pred_df):
    # SAVE FILE
    logging.info("PREDICTING: Saving file with PII ...")
    sub_df = pred_df.drop(columns=["token_string", "label_id"]) # remove extra columns
    sub_df.to_csv("submission.csv", index=False)

# remove all PII in text
def save_processed_data(words, pred_df):
    try:
        logging.info("PREDICTING: Saving processed data ...")

        # process the text
        post_processed_text = words.apply(lambda text: remove_pii(text, pred_df))

        # Save the processed text to a file
        post_processed_text.to_csv("processed_data.csv", index=False, header=False)

        logging.info("PREDICTING: Processed data saved successfully.")
    except Exception as e:
        logging.error(f"Error: {e}. Please check if the *save_processed_data() predict.py* exists and works properly.")
        raise


# Your prediction code here
def make_predictions(model, processed_test_data):
    try:
        try:
            logging.info("PREDICTING: Preparing data ...")
            words, docs, test_token_ids, test_ds = processed_test_data
        except FileNotFoundError as e:
            logging.error(f"Error: {e}. Please check if the *make_predictions() predict.py* and check if the specified file exists.")
            raise

        # INFERENCE
        logging.info("PREDICTING: Predicting the results ...")
        test_preds = model.predict(test_ds, verbose=1)

        # Convert probabilities to class labels via max confidence
        logging.info("PREDICTING: Converting probabilities into classes ...")
        test_preds = np.argmax(test_preds, axis=-1)

        # POST PROCESSING
        document_list = []
        token_id_list = []
        label_id_list = []
        token_list = []
        word_list = []

        logging.info("PREDICTING: Post processing data ... ")
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
                word_list.extend(tokens)

        # SUBMISSION
        pred_df = pd.DataFrame(
            {
                "document": document_list,
                "token": token_id_list,
                "label_id": label_id_list,
                "token_string": token_list,
                "words": word_list,
            }
        )
        pred_df = pred_df.rename_axis("row_id").reset_index() # add `row_id` column
        pred_df["label"] = pred_df.label_id.map(ModelConfiguration.id2label) # map integer label to BIO format label

        # save file with PII
        save_predictions(pred_df)

        # save text with no PII
        save_processed_data(words, pred_df)

        logging.info("PREDICTING: Done!")
    except Exception as e:
        logging.error(f"Error: Please check if the *make_predictions() predict.py* exists and work properly. \n{e}")
        raise