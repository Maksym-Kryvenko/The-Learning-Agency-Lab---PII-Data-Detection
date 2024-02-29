import logging

# Configure the logging module
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,                     # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
        handlers=[
            logging.StreamHandler(),            # Log to console
            logging.FileHandler('logfile.log')  # Log to a file
        ]
    )
    return logging

# Class to define input data paths
class InputData:
    train = "/input/train.json"
    test = "/input/test.json"
    sample = "/input/sample_submission.csv"
    trained_model = "/model.weights.h5"

# Configuration parameters for the model
class ModelConfiguration:
    seed = 2024
    preset = 'deberta_v3_small_en'  # pretrained model
    train_seq_len = 1024            # max size of input
    train_batch_size = 8            # size of input batch
    infer_seq_len = 2000            # max size of input sequence for inference
    infer_batch_size = 2            # size of input batch in inference
    epochs = 6                      # number of epochs to train
    lr_mode = 'exp'                 # or 'cos' or 'step'
    
    labels = ['B-USERNAME', 'B-ID_NUM', 'I-PHONE_NUM', 'I-ID_NUM', 'I-NAME_STUDENT', 'B-EMAIL', 'I-STREET_ADDRESS', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'O', 'I-URL_PERSONAL', 'B-PHONE_NUM', 'B-NAME_STUDENT']
    id2label = dict(enumerate(labels))  # integer values for BIO mapping
    label2id = {value:key for key,value in id2label.items()}
    num_labels = len(labels)        # number of PII tags
    
    train = False                   # whether to train or use already trained 

