import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# Load the BERT model and tokenizer
def load_model_and_tokenizer(model_name):
    print("Loading model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

# Load the dataset
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    data = pd.read_csv(file_path)
    return data

# Tokenize the dataset
def tokenize_dataset(data, tokenizer):
    inputs = tokenizer(data['statement'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs

# Make predictions
def make_predictions(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1].numpy()  # Get probabilities for the positive class
        predictions = torch.argmax(outputs.logits, dim=-1).numpy()
    return predictions, probabilities

# Calculate accuracy and AUC
def calculate_metrics(true_labels, predictions, probabilities):
    accuracy = accuracy_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    return accuracy, auc

# Main function
def main():
    model_name = 'bert-base-uncased'
    datasets_folder = 'datasets'

    model, tokenizer = load_model_and_tokenizer(model_name)

    for dataset_file in os.listdir(datasets_folder):
        if dataset_file.endswith(".csv"):
            data_file_path = os.path.join(datasets_folder, dataset_file)
            data = load_dataset(data_file_path)

            inputs = tokenize_dataset(data, tokenizer)
            predictions, probabilities = make_predictions(model, inputs)

            true_labels = data['label'].tolist()
            accuracy, auc = calculate_metrics(true_labels, predictions, probabilities)

            print(f"Accuracy for {dataset_file}: {accuracy:.5f}")
            print(f"AUC for {dataset_file}: {auc:.5f}\n")

if __name__ == "__main__":
    main()