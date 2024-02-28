from preprocessing import preprocess_data
from training import train_model
from predict import make_predictions
import keras



# all files in one class
class InputData:
    train = "/input/train.json"
    test = "/input/test.json"
    sample = "/input/sample_submission.csv"
    trained_model = "/input/model.weights.h5"

# configuration parameters
class ModelConfiguration:
    seed = 2024
    preset = 'deberta_v3_small_en'  # pretrained model
    train_seq_len = 1024       # max size of input
    train_batch_size = 8   # size of input batch
    infer_seq_len = 2000       # max size of input sequence for inference
    infer_batch_size = 2   # size of input batch in inference
    epochs = 6                 # number of epochs to train
    lr_mode = 'exp' # or 'cos' or 'step'
    
    labels = ['B-USERNAME', 'B-ID_NUM', 'I-PHONE_NUM', 'I-ID_NUM', 'I-NAME_STUDENT', 'B-EMAIL', 'I-STREET_ADDRESS', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'O', 'I-URL_PERSONAL', 'B-PHONE_NUM', 'B-NAME_STUDENT']
    id2label = dict(enumerate(labels))  # integer values for BIO mapping
    label2id = {value:key for key,value in id2label.items()}
    num_labels = len(labels)  # number of PII tags
    
    train = False  # whether to train or use already trained 


# Execute your code in the necessary order
if __name__ == "__main__":

    keras.utils.set_random_seed(ModelConfiguration.seed)      # produce similar result in each run
    keras.mixed_precision.set_global_policy("mixed_float16")  # enable larger batch sizes and faster training

    processed_train_data = preprocess_data(InputData.train)
    trained_model = train_model(processed_train_data, InputData)
    processed_test_data = preprocess_data(InputData.test)
    predictions = make_predictions(trained_model, processed_test_data)
    