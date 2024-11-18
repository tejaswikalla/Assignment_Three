import torch
from transformers import AutoTokenizer, OPTForCausalLM, LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='embedding_extraction.log')

def load_llama_model(model_name: str):
    """
    Initializes and returns a LLaMa model and tokenizer.
    """
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the model name
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_data(dataset_path: Path, dataset_name: str, true_false: bool = False):
    """
    Loads the dataset from a CSV file.
    """
    filename_suffix = "_true_false" if true_false else ""
    dataset_file = dataset_path / f"{dataset_name}{filename_suffix}.csv"
    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError as e:
        print(f"Dataset file {dataset_file} not found: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file {dataset_file}: {str(e)}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"No data in CSV file {dataset_file}: {str(e)}")
        return None
    if 'embeddings' not in df.columns:
        df['embeddings'] = pd.Series(dtype='object')
    return df

def process_row(prompt: str, model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Processes a row of data and returns the embeddings.
    """
    if remove_period:
        prompt = prompt.rstrip(". ")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
    embeddings = {}
    for layer in layers_to_use:
        last_hidden_state = outputs.hidden_states[0][layer][0][-1]
        embeddings[layer] = [last_hidden_state.numpy().tolist()]
    return embeddings

def process_batch(batch_prompts: List[str], model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Processes a batch of data and returns the embeddings for each statement.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    if remove_period:
        batch_prompts = [prompt.rstrip(". ") for prompt in batch_prompts]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

    # Use the attention mask to find the index of the last real token for each sequence
    seq_lengths = inputs.attention_mask.sum(dim=1) - 1  # Subtract 1 to get the index

    batch_embeddings = {}
    for layer in layers_to_use:
        hidden_states = outputs.hidden_states[layer]

        # Gather the hidden state at the last real token for each sequence
        last_hidden_states = hidden_states[range(hidden_states.size(0)), seq_lengths, :]
        batch_embeddings[layer] = [embedding.detach().cpu().numpy().tolist() for embedding in last_hidden_states]

    return batch_embeddings

def save_data(df, output_path: Path, dataset_name: str, model_name: str, layer: int, remove_period: bool):
    """
    Saves the processed data to a CSV file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_rmv_period" if remove_period else ""
    output_file = output_path / f"embeddings_{dataset_name}{model_name}_{abs(layer)}{filename_suffix}.csv"
    try:
        df.to_csv(output_file, index=False)
    except PermissionError:
        print(f"Permission denied when trying to write to {output_file}. Please check your file permissions.")
    except Exception as e:
        print(f"An unexpected error occurred when trying to write to {output_file}: {e}")

def main():
    """
    Main function to load configuration parameters, initialize the model and tokenizer, and process datasets.
    """
    try:
        with open("config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found. Please ensure the file exists and the path is correct.")
        return
    except PermissionError:
        logging.error("Permission denied. Please check your file permissions.")
        return
    except json.JSONDecodeError:
        logging.error("Configuration file is not valid JSON. Please check the file's contents.")
        return

    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--model", help="Name of the language model to use: '6.7b', '2.7b', '1.3b', '350m'")
    parser.add_argument("--layers", nargs='*', help="List of layers of the LM to save embeddings from indexed negatively from the end")
    parser.add_argument("--dataset_names", nargs='*', help="List of dataset names without csv extension. Can leave off 'true_false' suffix if true_false flag is set to True")
    parser.add_argument("--true_false", action="store_true", help="Do you want to append 'true_false' to the dataset name?")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.")
    parser.add_argument("--remove_period", action="store_true", help="Include this flag if you want to extract embedding for the last token before the final period.")
    args = parser.parse_args()

    model_name = config_parameters["model"]
    should_remove_period = config_parameters["remove_period"]
    layers_to_process = config_parameters["layers_to_use"]
    dataset_names = config_parameters["list_of_datasets"]
    true_false = config_parameters["true_false"]
    BATCH_SIZE = config_parameters["batch_size"]
    dataset_path = Path(config_parameters["dataset_path"])
    output_path = Path(config_parameters["processed_dataset_path"])

    model_output_per_layer: Dict[int, pd.DataFrame] = {}

    if '2-7b-hf' in model_name:
        model, tokenizer = load_llama_model(model_name)
        
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer initialization failed.")
        return
  
    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        np.set_printoptions(threshold=np.inf)
        dataset = load_data(dataset_path, dataset_name, true_false=true_false)
        if dataset is None:
            continue

        num_batches = len(dataset) // BATCH_SIZE + (len(dataset) % BATCH_SIZE != 0)

        for layer in layers_to_process:
            model_output_per_layer[layer] = dataset.copy()
            model_output_per_layer[layer]['embeddings'] = pd.Series(dtype='object')

        for batch_num in tqdm(range(num_batches), desc=f"Processing batches in {dataset_name}"):
            start_idx = batch_num * BATCH_SIZE
            actual_batch_size = min(BATCH_SIZE, len(dataset) - start_idx)
            end_idx = start_idx + actual_batch_size
            batch = dataset.iloc[start_idx:end_idx]
            batch_prompts = batch['statement'].tolist()
            batch_embeddings = process_batch(batch_prompts, model, tokenizer, layers_to_process, should_remove_period)

            for layer in layers_to_process:
                for i, idx in enumerate(range(start_idx, end_idx)):
                    model_output_per_layer[layer].at[idx, 'embeddings'] = batch_embeddings[layer][i]

            if batch_num % 10 == 0:
                logging.info(f"Processing batch {batch_num}")

        for layer in layers_to_process:
            save_data(model_output_per_layer[layer], output_path, dataset_name, model_name, layer, should_remove_period)

if __name__ == "__main__":
    main()