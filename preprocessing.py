import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import tensorflow as tf
import keras
import keras_nlp
from keras import ops
from config import ModelConfiguration, configure_logging


logging = configure_logging()  # Configure logging from config.py

# read data 
def read_data(data_path):
    try:
        # Initialize empty arrays
        logging.info("PREPROCESSING: Reading data ...")
        data = json.load(open(data_path))

        words = np.empty(len(data), dtype=object)
        labels = np.empty(len(data), dtype=object)

        # Fill the arrays
        for i, x in tqdm(enumerate(data), total=len(data)):
            words[i] = np.array(x["tokens"])
            labels[i] = np.array([ModelConfiguration.label2id[label] for label in x["labels"]])

        return words, labels
    except Exception as e:
        logging.error(f"Error: Please check if the *read_data() preprocessing.py* exists and work properly. \n{e}")
        raise

# create tokenizer
def tokenizer_conf():
    try:
        # To convert string input or list of strings input to numerical tokens
        logging.info("PREPROCESSING: Creating tokenizer ...")
        tokenizer = keras_nlp.models.DebertaV3Tokenizer.from_preset(
            ModelConfiguration.preset,
        )
        return tokenizer
    except Exception as e:
        logging.error(f"Error: Please check if the *tokenizer_conf() preprocessing.py* exists and work properly. \n{e}")
        raise

# create packer
def packer_conf():
    try:    
        tokenizer = tokenizer_conf()
        # Preprocessing layer to add spetical tokens: [CLS], [SEP], [PAD]
        logging.info("PREPROCESSING: Creating packer ...")
        packer = keras_nlp.layers.MultiSegmentPacker(
            start_value=tokenizer.cls_token_id,
            end_value=tokenizer.sep_token_id,
            sequence_length=10,
        )
        return packer
    except Exception as e:
        logging.error(f"Error: Please check if the *packer_conf() preprocessing.py* exists and work properly. \n{e}")
        raise


# get tokens from words
def get_tokens(words, seq_len):
    try:
        logging.info("PREPROCESSING: Getting tokens ...")
        tokenizer = tokenizer_conf()
        packer = packer_conf()
        # Tokenize input
        token_words = tf.expand_dims(
            tokenizer(words), axis=-1
        )  # ex: (words) ["It's", "a", "cat"] ->  (token_words) [[1, 2], [3], [4]]
        tokens = tf.reshape(
            token_words, [-1]
        )  # ex: (token_words) [[1, 2], [3], [4]] -> (tokens) [1, 2, 3, 4]
        # Pad tokens
        tokens = packer(tokens)[0][:seq_len]
        inputs = {"token_ids": tokens, "padding_mask": tokens != 0}
        return inputs, tokens, token_words
    except Exception as e:
        logging.error(f"Error: Please check if the *get_tokens() preprocessing.py* exists and work properly. \n{e}")
        raise

# get token ids for tokens
def get_token_ids(token_words):
    try:
        logging.info("PREPROCESSING: Getting token ids ...")
        # Get word indices
        word_ids = tf.range(tf.shape(token_words)[0])
        # Get size of each word
        word_size = tf.reshape(tf.map_fn(lambda word: tf.shape(word)[0:1], token_words), [-1])
        # Repeat word_id with size of word to get token_id
        token_ids = tf.repeat(word_ids, word_size)
        return token_ids
    except Exception as e:
        logging.error(f"Error: Please check if the *get_token_ids() preprocessing.py* exists and work properly. \n{e}")
        raise

# get token labels
def get_token_labels(word_labels, token_ids, seq_len):
    try:
        logging.info("PREPROCESSING: Getting token labels ...")
        # Create token_labels from word_labels ->  alignment
        token_labels = tf.gather(word_labels, token_ids)
        # Only label the first token of a given word and assign -100 to others
        mask = tf.concat([[True], token_ids[1:] != token_ids[:-1]], axis=0)
        token_labels = tf.where(mask, token_labels, -100)
        # Truncate to max sequence length
        token_labels = token_labels[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
        # Pad token_labels to align with tokens (use -100 to pad for loss/metric ignore)
        pad_start = 1  # for [CLS] token
        pad_end = seq_len - tf.shape(token_labels)[0] - 1  # for [SEP] and [PAD] tokens
        token_labels = tf.pad(token_labels, [[pad_start, pad_end]], constant_values=-100)
        return token_labels
    except Exception as e:
        logging.error(f"Error: Please check if the *get_token_labels() preprocessing.py* exists and work properly. \n{e}")
        raise

# add special tokens to define start and end of the sentance
def process_token_ids(token_ids, seq_len):
    try:    
        logging.info("PREPROCESSING: Processing token ids ...")
        # Truncate to max sequence length
        token_ids = token_ids[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
        # Pad token_ids to align with tokens (use -1 to pad for later identification)
        pad_start = 1  # [CLS] token
        pad_end = seq_len - tf.shape(token_ids)[0] - 1  # [SEP] and [PAD] tokens
        token_ids = tf.pad(token_ids, [[pad_start, pad_end]], constant_values=-1)
        return token_ids
    except Exception as e:
        logging.error(f"Error: Please check if the *process_token_ids() preprocessing.py* exists and work properly. \n{e}")
        raise

# tokenize text
def process_data(seq_len=720, has_label=True, return_ids=False):
    try:    
        tokenizer = tokenizer_conf()
        # To add spetical tokens: [CLS], [SEP], [PAD]
        packer = keras_nlp.layers.MultiSegmentPacker(
            start_value=tokenizer.cls_token_id,
            end_value=tokenizer.sep_token_id,
            sequence_length=seq_len,
        )

        def process(x):
            try:    
                logging.info("PREPROCESSING: Processing data ...")
                # Generate inputs from tokens
                inputs, tokens, words_int = get_tokens(x["words"], seq_len, packer)
                # Generate token_ids for maping tokens to words
                token_ids = get_token_ids(words_int)
                if has_label:
                    # Generate token_labels from word_labels
                    token_labels = get_token_labels(x["labels"], token_ids, seq_len)
                    return inputs, token_labels
                elif return_ids:
                    # Pad token_ids to align with tokens
                    token_ids = process_token_ids(token_ids, seq_len)
                    return token_ids
                else:
                    return inputs
            except Exception as e:
                logging.error(f"Error: Please check if the *process() into process_data() preprocessing.py* exists and work properly. \n{e}")
                raise

        return process
    except Exception as e:
        logging.error(f"Error: Please check if the *process_data() preprocessing.py* exists and work properly. \n{e}")
        raise

# DATA LOADING FUNCTION
def build_dataset(words, labels=None, return_ids=False, batch_size=4,
                  seq_len=512, shuffle=False, cache=True, drop_remainder=True):
    try:
        logging.info("PREPROCESSING: Building dataset ...")
        AUTO = tf.data.AUTOTUNE 

        slices = {"words": tf.ragged.constant(words)}
        if labels is not None:
            slices.update({"labels": tf.ragged.constant(labels)})

        ds = tf.data.Dataset.from_tensor_slices(slices)
        ds = ds.map(process_data(seq_len=seq_len,
                                has_label=labels is not None, 
                                return_ids=return_ids), num_parallel_calls=AUTO) # apply processing
        ds = ds.cache() if cache else ds  # cache dataset
        if shuffle: # shuffle dataset
            ds = ds.shuffle(1024, seed=ModelConfiguration.seed)  
            opt = tf.data.Options() 
            opt.experimental_deterministic = False
            ds = ds.with_options(opt)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)  # batch dataset
        ds = ds.prefetch(AUTO)  # prefetch next batch
        return ds
    except Exception as e:
        logging.error(f"Error: Please check if the *build_dataset() preprocessing.py* exists and work properly. \n{e}")
        raise


# Your data preparation code here
def preprocess_data(data_path):
    try:
        # Set random seed and global policy for reproducibility
        keras.utils.set_random_seed(ModelConfiguration.seed)      # produce similar result in each run
        keras.mixed_precision.set_global_policy("mixed_float16")  # enable larger batch sizes and faster training

        # read data
        words, labels = read_data(data_path, ModelConfiguration)

        if ModelConfiguration.train:
            # split the data
            train_tokens, valid_tokens, train_labels, valid_labels = train_test_split(
                words,
                labels,
                test_size = 0.3
            )

            # BUILD TRAIN AND VALID DATALOADER
            train_ds = build_dataset(train_tokens, train_labels,  batch_size=ModelConfiguration.train_batch_size,
                                    seq_len=ModelConfiguration.train_seq_len, shuffle=True)

            valid_ds = build_dataset(valid_tokens, valid_labels, batch_size=ModelConfiguration.train_batch_size, 
                                    seq_len=ModelConfiguration.train_seq_len, shuffle=False)
            return train_ds, valid_ds
        
        else:
            # Get token ids
            id_ds = build_dataset(words, return_ids=True, batch_size=len(words), 
                                    seq_len=ModelConfiguration.infer_seq_len, shuffle=False, cache=False, drop_remainder=False)
            test_token_ids = ops.convert_to_numpy([ids for ids in iter(id_ds)][0])

            # Build test dataloader
            test_ds = build_dataset(words, return_ids=False, batch_size=ModelConfiguration.infer_batch_size,
                                    seq_len=ModelConfiguration.infer_seq_len, shuffle=False, cache=False, drop_remainder=False)
            return words, labels, test_token_ids, test_ds
    except Exception as e:
        logging.error(f"Error: Please check if the *preprocess_data() preprocessing.py* exists and work properly. \n{e}")
        raise
