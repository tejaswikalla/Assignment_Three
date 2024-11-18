
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pathlib import Path
import json
import logging
import os
from copy import deepcopy
import sys
import argparse

def load_config(config_file):
    # Load the configuration settings from a JSON file.
    try:
        with open(config_file) as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"Config file {config_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {config_file}.")
        sys.exit(1)

def load_datasets(dataset_names, layers_to_process, should_remove_period, input_path, model_name, idx):
    # Load datasets based on the provided parameters.
    datasets = []
    dataset_paths = []
    for dataset_name in dataset_names:
        try:
            if should_remove_period:
                path = input_path / f"embeddings_{dataset_name}{model_name}_{abs(layers_to_process[idx])}_rmv_period.csv"
            else:
                path = input_path / f"embeddings_{dataset_name}{model_name}_{abs(layers_to_process[idx])}.csv"
            datasets.append(pd.read_csv(path))
            dataset_paths.append(path)
        except FileNotFoundError:
            print(f"File not found: {path}. Please ensure the dataset file exists.")
            sys.exit(1)
        except pd.errors.ParserError:
            print(f"Error parsing CSV file: {path}. Please ensure the file is in correct CSV format.")
            sys.exit(1)
    return datasets, dataset_paths

def prepare_datasets(datasets, dataset_names, test_first_only):
    # Prepare train and test datasets from a list of datasets.
    if not datasets or not dataset_names:
        raise ValueError("Both 'datasets' and 'dataset_names' must be nonempty.")
    if len(datasets) != len(dataset_names):
        raise ValueError("'datasets' and 'dataset_names' must have the same length.")
    
    train_datasets = []
    test_datasets = []
    dataset_loop_length = 1 if test_first_only else len(dataset_names)
    for ds in range(dataset_loop_length):
        if test_first_only:
            test_df = datasets[0]
            dfs_to_concatenate = datasets[1:]  
        else:
            test_df = datasets[ds]
            dfs_to_concatenate = datasets[:ds] + datasets[ds + 1:]
        
        train_df = pd.concat(dfs_to_concatenate, ignore_index=True)
        train_datasets.append(train_df)
        test_datasets.append(test_df)
    return train_datasets, test_datasets

def correct_str(str_arr):
    # Convert a string representation of a numpy array into a comma-separated string.
    val_to_ret = (str_arr.replace("[array(", "")
                        .replace("dtype=float32)]", "")
                        .replace("\n","")
                        .replace(" ","")
                        .replace("],","]")
                        .replace("[","")
                        .replace("]",""))
    return val_to_ret

def define_model(input_dim):
    # Define and compile a Sequential model in Keras.
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("Input dimension must be a positive integer.")
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_embeddings, train_labels):
    # Train the input model on the provided embeddings and labels.
    if len(train_embeddings) != len(train_labels):
        raise ValueError("Training embeddings and labels must have the same length.")
    model.fit(train_embeddings, train_labels, epochs=5, batch_size=32)
    return model

def evaluate_model(model, test_embeddings, test_labels):
    # Evaluate the performance of the trained model on the test data.
    if len(test_embeddings) != len(test_labels):
        raise ValueError("Test embeddings and labels must have the same length.")
    loss, accuracy = model.evaluate(test_embeddings, test_labels)
    return loss, accuracy

def compute_roc_curve(test_labels, test_pred_prob):
    # Compute the Receiver Operating Characteristic (ROC) curve and the area under the curve (AUC).
    fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)  # Assuming binary classification
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def find_optimal_threshold(X, y, model):
    # Find the optimal threshold for a binary classification model by maximizing the accuracy score over the ROC curve thresholds.
    y_pred_prob = model.predict(X)
    fpr_val, tpr_val, thresholds_val = roc_curve(y, y_pred_prob)
    optimal_threshold = thresholds_val[np.argmax([accuracy_score(y, y_pred_prob > thr) for thr in thresholds_val])]
    return optimal_threshold

def print_results(results, dataset_names, repeat_each, layer_num_from_end):
    # Print the average accuracy, AUC, and optimal threshold for each dataset, and return a list of these results.
    if len(results) != len(dataset_names) * repeat_each:
        raise ValueError("Results array length should be equal to dataset_names length multiplied by repeat_each.")
    overall_res = []
    for ds in range(len(dataset_names)):
        relevant_results_portion = results[repeat_each*ds:repeat_each*(ds+1)]
        acc_list = [t[2] for t in relevant_results_portion]
        auc_list = [t[3] for t in relevant_results_portion]
        opt_thresh_list = [t[4] for t in relevant_results_portion]
        avg_acc = sum(acc_list) / len(acc_list)
        avg_auc = sum(auc_list) / len(auc_list)
        avg_thrsh = sum(opt_thresh_list) / len(opt_thresh_list)
        text_res = ("dataset: " + str(dataset_names[ds]) + " layer_num_from_end:" 
                    + str(layer_num_from_end) + " Avg_acc:" + str(avg_acc) 
                    + " Avg_AUC:" + str(avg_auc) + " Avg_threshold:" 
                    + str(avg_thrsh))
        print(text_res)
        overall_res.append(text_res)
    return overall_res

def main():
    # Define the logger
    try:
        logging.basicConfig(filename='classification.log', level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return
    
    logger.info("Execution started.")

    # Load the config
    try:
        config_parameters = load_config("config.json")
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return

    parser = argparse.ArgumentParser(description="Train probes on processed and labeled datasets.")
    parser.add_argument("--model", help="Name of the language model to use: '6.7b', '2.7b', '1.3b', '350m'")
    parser.add_argument("--layers", nargs='*', help="List of layers of the LM to save embeddings from indexed negatively from the end")
    parser.add_argument("--dataset_names", nargs='*', help="List of dataset names without csv extension.")
    parser.add_argument("--remove_period", type=bool, help="True if you want to extract embedding for last token before the final period.")
    parser.add_argument("--test_first_only", type=bool, help="True if you only want to use the first dataset for testing a probe.")
    parser.add_argument("--save_probes", type=bool, help="True if you want to save the trained probes.")
    parser.add_argument("--repeat_each", type=int, help="How many times to train a randomly initialized probe for each dataset.")
    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    should_remove_period = args.remove_period if args.remove_period is not None else config_parameters["remove_period"]
    layers_to_process = [int(x) for x in args.layers] if args.layers is not None else config_parameters["layers_to_use"]
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters["list_of_datasets"]
    test_first_only = args.test_first_only if args.test_first_only is not None else config_parameters["test_first_only"]
    save_probes = args.save_probes if args.save_probes is not None else config_parameters["save_probes"]
    repeat_each = args.repeat_each if args.repeat_each is not None else config_parameters["repeat_each"]
    input_path = Path(config_parameters["processed_dataset_path"])
    probes_path = Path(config_parameters["probes_dir"])
    
    overall_res = []

    # Iterate over the layers in "layer_num_list"
    for idx in range(len(layers_to_process)):
        # Load the datasets
        datasets, dataset_paths = load_datasets(dataset_names, layers_to_process, should_remove_period, input_path, model_name, idx)

        # Prepare the datasets
        train_datasets, test_datasets = prepare_datasets(datasets, dataset_names, test_first_only)

        results = []
 
        # Iterate over each test dataset
        for count, (test_dataset, train_dataset, test_dataset_path) in enumerate(zip(test_datasets, train_datasets, dataset_paths)):

            # Split the dataset into embeddings and labels
            train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_dataset['embeddings'].tolist()])
            train_labels = train_dataset['label']
            test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_dataset['embeddings'].tolist()])
            test_labels = test_dataset['label']

            # Repeat training and testing for specified number of times
            best_accuracy = 0
            best_model = None
            all_probs_list = []
            for i in range(repeat_each):

                # Define the model
                model = define_model(train_embeddings.shape[1])
            
                # Train the model on full training data
                model = train_model(model, train_embeddings, train_labels)

                # Find the optimal threshold and compute validation set accuracy
                optimal_threshold = find_optimal_threshold(train_embeddings, train_labels, model)

                # Evaluate the model
                loss, accuracy = evaluate_model(model, test_embeddings, test_labels)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

                test_pred_prob = model.predict(test_embeddings)
                all_probs_list.append(deepcopy(test_pred_prob)) # Store probabilities

                # Compute ROC curve and ROC area
                roc_auc, fpr, tpr = compute_roc_curve(test_labels, model.predict(test_embeddings))

                # Compute test set accuracy using the optimal threshold
                test_accuracy = accuracy_score(test_labels, model.predict(test_embeddings) > optimal_threshold)

                results.append((dataset_names[count], i, accuracy, roc_auc, optimal_threshold, test_accuracy))

            # Save the best model
            if save_probes:
                try:
                    if not os.path.exists(probes_path):
                        os.makedirs(probes_path)
                    if should_remove_period:
                        model_path = os.path.join(probes_path, f"{model_name}_{abs(layers_to_process[idx])}_{dataset_names[count]}_rp.h5")
                    else:
                        model_path = os.path.join(probes_path, f"{model_name}_{abs(layers_to_process[idx])}_{dataset_names[count]}.h5")
                    best_model.save(model_path)
                except Exception as e:
                    logger.error(f"Error saving the model: {e}")

            # Add probabilities to the CSV file
            test_dataset_copy = test_dataset.copy()  # Make a copy of the dataset
            test_dataset_copy['average_probability'] = np.mean(all_probs_list, axis=0)
            for i, prob in enumerate(all_probs_list):
                test_dataset_copy[f'model_{i}_probability'] = prob
            # Round off the probabilities to the first four digits
            for col in test_dataset_copy.columns:
                if 'probability' in col:
                    test_dataset_copy[col] = test_dataset_copy[col].round(4)

            # Define the new filename for the copy
            original_path = Path(test_dataset_path)
            new_path = original_path.with_name(original_path.name.replace('.csv', '_predictions.csv'))

            # Save the modified copy as a new CSV file
            test_dataset_copy.to_csv(new_path, index=False) 

        # Print the results
        if not test_first_only:
            avg_res = print_results(results, dataset_names, repeat_each, layers_to_process[idx])
            overall_res.extend(avg_res)

    logger.info("Execution completed.")
    logger.info("Overall results: " + str(overall_res))

if __name__ == "__main__":
    main()