from preprocessing import preprocess_data
from training import train_model
from predict import make_predictions
from config import InputData


# Execute code in the necessary order
def main():
    # Preprocess training data
    processed_train_data = preprocess_data(InputData.train)

    # Train or load a pre-trained model depends on "train" parameter in config.py
    trained_model = train_model(processed_train_data, InputData)
    
    # Preprocess test data and make predictions
    processed_test_data = preprocess_data(InputData.test)
    predictions = make_predictions(trained_model, processed_test_data)


# Execute code in the necessary order
if __name__ == "__main__":
    main()
