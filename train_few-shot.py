import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load the model and tokenizer
def load_model_and_tokenizer(model_name):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = OPTForCausalLM.from_pretrained(model_name)
    model.half()  # Use mixed precision
    return model, tokenizer

# Load the dataset from a CSV file
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    data = pd.read_csv(file_path)
    return data

# Create a few-shot prompt from the dataset
def create_few_shot_prompt(data, n_shots):
    print(f"Creating {n_shots}-shot prompt...")
    few_shot_examples = data.sample(n=n_shots)
    few_shot_prompt = ""
    for _, row in few_shot_examples.iterrows():
        label = "True" if row['label'] == 1 else "False"
        few_shot_prompt += f"Input: {row['statement']}\nOutput: {label}\n\n"
    return few_shot_prompt

# Custom Dataset class for handling statements
class StatementDataset(Dataset):
    def __init__(self, statements, few_shot_prompt):
        self.statements = statements
        self.few_shot_prompt = few_shot_prompt

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        statement = self.statements[idx]
        prompt = self.few_shot_prompt + f"Input: {statement}\nOutput:"
        return prompt

# Classify a batch of statements
def classify_statements(prompts, tokenizer, model, temperature=0.7, top_k=10, max_length=50):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)

    with autocast():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 10,
            temperature=temperature,
            top_k=top_k,
            do_sample=True
        )

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predicted_labels = [prediction.split("Output:")[-1].strip().lower() for prediction in predictions]
    return ["True" if "true" in label else "False" for label in predicted_labels]

# Evaluate the model using ROC AUC and accuracy
def evaluate_model(model, tokenizer, dataset, few_shot_prompt, batch_size=5):
    true_labels = dataset['label'].tolist()
    statements = dataset['statement'].tolist()

    statement_dataset = StatementDataset(statements, few_shot_prompt)
    dataloader = DataLoader(statement_dataset, batch_size=batch_size, num_workers=4)

    predicted_labels = []
    for batch_prompts in tqdm(dataloader, desc="Classifying statements"):
        predicted_label = classify_statements(batch_prompts, tokenizer, model)
        predicted_labels.extend(predicted_label)

    predicted_probs = [1.0 if label.lower() == "true" else 0.0 for label in predicted_labels]

    accuracy = calculate_accuracy(true_labels, predicted_probs)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
   
    return accuracy, roc_auc

# Calculate accuracy
def calculate_accuracy(true_labels, predicted_labels):
    return accuracy_score(true_labels, predicted_labels)

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism
    set_seed(42)

    model_name = "facebook/opt-6.7b"
    datasets_folder = "datasets"

    model, tokenizer = load_model_and_tokenizer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("Hello")

    for dataset_file in os.listdir(datasets_folder):
        if dataset_file.endswith(".csv"):
            data_file_path = os.path.join(datasets_folder, dataset_file)
            data = load_dataset(data_file_path)

            few_shot_prompt = create_few_shot_prompt(data, n_shots=3)

            print(f"Processing statements for {dataset_file}...")
            accuracy, roc_auc = evaluate_model(model, tokenizer, data, few_shot_prompt)

            print(f"Accuracy for {dataset_file}: {accuracy:.4f}%")
            print(f"Avg_acc: {accuracy:.4f}% Avg_AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
