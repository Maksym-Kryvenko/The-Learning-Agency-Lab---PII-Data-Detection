# Personally identifiable information (PII) detection

## Project Overview
This project aims to perform Named Entity Recognition (NER) on text data and remove personally identifiable information (PII) entities from the text. The NER model is trained using a Deep Learning approach, and spaCy is utilized for identifying and removing PII entities.

## Acknowledgments
- This project was inspired by the need for efficient PII removal techniques in natural language processing applications.
- I thank the open-source community for their contributions to libraries such as TensorFlow, Keras, spaCy, and pandas, which made this project possible. Also, I highly apreciate the possibility of free usage of [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) for such educational purposes.
- This project based on task and dataset from [The-Learning-Agency-Lab-PII-Data-Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data).

## Project Structure

The project is organized into three main files:

1. **preprocessing.py:** Handles data preprocessing tasks, including reading data, tokenization, and building datasets for training and testing.

2. **training.py:** Contains code for model training. Defines the neural network architecture, loss functions, and metrics. Also includes learning rate scheduling.

3. **predict.py:** Implements the prediction pipeline. Preprocesses the test data, loads the trained model, and makes predictions. It also post-processes the predictions, removing personally identifiable information (PII) entities.

4. **logfile.log** Keep track of all processes and record process history and errors.

5. **main.py** Main file that execute all other processes.

6. **exploration_notebook.ipynb** Notebook with data exploration, comments and project pipeline. 

## File Descriptions

- **config.py:** Centralized configuration file for model and training parameters.
- **requirements.txt:** List of all required modules and libraries.

## Dependencies

Ensure you have the necessary dependencies installed. You can install them using the following:

```bash
pip install -r requirements.txt
```

## Setup

1. **Data Preparation:**
   - Place your training and testing data in the appropriate folders.
   - Modify paths and configurations in `config.py` to match your dataset.

2. **Training:**
   - Run `training.py` to train the model.
   - There are two options either to train model or use pre-trained one. Choose it in `config.py`.

3. **Prediction:**
   - After training, run `predict.py` to make predictions on the test set.

## Results

- The final predictions are saved in `submission.csv`.
- Processed data (text without PII) is saved in `processed_data.csv`.

## Usage

- Modify the configurations in `config.py` based on your dataset and requirements.
- Ensure the necessary data is available and organized.
- Run the preprocessing, training, and prediction scripts sequentially.

## Notes

- The project uses [SpaCy](https://spacy.io/) for named entity recognition during post-processing.
- Adjust the SpaCy model based on specific entity requirements.
- Due to storage limitations on GitHub it is not possible to save trained model file here, so contact me if you need to get it.

## Author

[Maksym Kryvenko]

Feel free to reach out for any questions or improvements!


