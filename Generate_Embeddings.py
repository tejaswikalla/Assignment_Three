import torch
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='embedding_extraction.log')

def initialize_opt_model(model_name: str):
    """Load the OPT model and tokenizer based on the model name."""
    try:
        model = OPTForCausalLM.from_pretrained(f"facebook/opt-{model_name}")
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{model_name}")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return None, None
    return model, tokenizer

def read_dataset(dataset_path: Path, dataset_name: str, true_false: bool = False):
    """Read the dataset from a CSV file and return a DataFrame."""
    suffix = "_true_false" if true_false else ""
    file_path = dataset_path / f"{dataset_name}{suffix}.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {str(e)}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"Empty CSV file: {str(e)}")
        return None
    if 'embeddings' not in df.columns:
        df['embeddings'] = pd.Series(dtype='object')
    return df

def process_single_row(prompt: str, model, tokenizer, layers: list, remove_period: bool):
    """Process a single row and return the embeddings."""
    if remove_period:
        prompt = prompt.rstrip(". ")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
    embeddings = {}
    for layer in layers:
        last_hidden_state = outputs.hidden_states[0][layer][0][-1]
        embeddings[layer] = [last_hidden_state.numpy().tolist()]
    return embeddings

def process_multiple_rows(prompts: List[str], model, tokenizer, layers: list, remove_period: bool):
    """Process multiple rows and return the embeddings for each statement."""
    if remove_period:
        prompts = [prompt.rstrip(". ") for prompt in prompts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    seq_lengths = inputs.attention_mask.sum(dim=1) - 1

    batch_embeddings = {}
    for layer in layers:
        hidden_states = outputs.hidden_states[layer]
        last_hidden_states = hidden_states[range(hidden_states.size(0)), seq_lengths, :]
        batch_embeddings[layer] = [embedding.detach().cpu().numpy().tolist() for embedding in last_hidden_states]

    return batch_embeddings

def save_embeddings(df, output_path: Path, dataset_name: str, model_name: str, layer: int, remove_period: bool):
    """Save the DataFrame with embeddings to a CSV file."""
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = "_rmv_period" if remove_period else ""
    if 'llama2-7b' in model_name:
        model_name = 'llama2-7b'
    output_file = output_path / f"embeddings_{dataset_name}{model_name}_{abs(layer)}{suffix}.csv"
    try:
        df.to_csv(output_file, index=False)
    except PermissionError:
        print(f"Permission denied: {output_file}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Main function to load configuration, initialize models, and process datasets."""
    try:
        with open("config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return
    except PermissionError:
        logging.error("Permission denied.")
        return
    except json.JSONDecodeError:
        logging.error("Invalid JSON in configuration file.")
        return

    parser = argparse.ArgumentParser(description="Generate new CSV with embeddings.")
    parser.add_argument("--model", help="Name of the language model to use: '6.7b', '2-7b-hf'")
    parser.add_argument("--layers", nargs='*', help="List of layers to save embeddings from, indexed negatively from the end")
    parser.add_argument("--dataset_names", nargs='*', help="List of dataset names without CSV extension.")
    parser.add_argument("--true_false", type=bool, help="Append 'true_false' to the dataset name?")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.")
    parser.add_argument("--remove_period", type=bool, help="Remove period before extracting embedding?")
    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    remove_period = args.remove_period if args.remove_period is not None else config_parameters["remove_period"]
    layers = [int(x) for x in args.layers] if args.layers is not None else config_parameters["layers_to_use"]
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters["list_of_datasets"]
    true_false = args.true_false if args.true_false is not None else config_parameters["true_false"]
    batch_size = args.batch_size if args.batch_size is not None else config_parameters["batch_size"]
    dataset_path = Path(config_parameters["dataset_path"])
    output_path = Path(config_parameters["processed_dataset_path"])

    model_output_per_layer: Dict[int, pd.DataFrame] = {}

    model, tokenizer = initialize_opt_model(model_name)
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer initialization failed.")
        return

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        np.set_printoptions(threshold=np.inf)
        dataset = read_dataset(dataset_path, dataset_name, true_false=true_false)
        if dataset is None:
            continue

        num_batches = len(dataset) // batch_size + (len(dataset) % batch_size != 0)

        for layer in layers:
            model_output_per_layer[layer] = dataset.copy()
            model_output_per_layer[layer]['embeddings'] = pd.Series(dtype='object')

        for batch_num in tqdm(range(num_batches), desc=f"Processing batches in {dataset_name}"):
            start_idx = batch_num * batch_size
            actual_batch_size = min(batch_size, len(dataset) - start_idx)
            end_idx = start_idx + actual_batch_size
            batch = dataset.iloc[start_idx:end_idx]
            batch_prompts = batch['statement'].tolist()
            batch_embeddings = process_multiple_rows(batch_prompts, model, tokenizer, layers, remove_period)

            for layer in layers:
                for i, idx in enumerate(range(start_idx, end_idx)):
                    model_output_per_layer[layer].at[idx, 'embeddings'] = batch_embeddings[layer][i]

            if batch_num % 10 == 0:
                logging.info(f"Processing batch {batch_num}")

        for layer in layers:
            save_embeddings(model_output_per_layer[layer], output_path, dataset_name, model_name, layer, remove_period)

if __name__ == "__main__":
    main()