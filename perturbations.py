import os
import pandas as pd
import random
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK punkt tokenizer
nltk.download('punkt', quiet=True)

def generate_one_perturbation(statement):
    """
    Generate a single perturbation for a given statement.
    """
    def introduce_typo(text):
        words = word_tokenize(text)
        word_to_change = random.choice(words)
        typo_word = ''.join(random.sample(word_to_change, len(word_to_change)))
        return ' '.join([typo_word if w == word_to_change else w for w in words])
    
    def change_word_order(text):
        words = word_tokenize(text)
        random.shuffle(words)
        return ' '.join(words)
    
    def add_noise(text):
        words = word_tokenize(text)
        noise_words = ['um', 'uh', 'like', 'you know']
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, random.choice(noise_words))
        return ' '.join(words)

    # Randomly choose one perturbation type
    perturbation_functions = [introduce_typo, change_word_order, add_noise]
    func = random.choice(perturbation_functions)
    
    return func(statement)

def create_perturbed_datasets(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # Create a list to store perturbed data
            perturbed_data = []

            # Generate one perturbation for each statement
            for _, row in df.iterrows():
                # Generate one perturbed statement
                perturbed_stmt = generate_one_perturbation(row['statement'])
                
                perturbed_data.append({
                    'statement': perturbed_stmt,
                    'label': row['label']  # Keep the original label
                })
            
            # Create a new DataFrame with only perturbed data
            perturbed_df = pd.DataFrame(perturbed_data)

            # Save to a new CSV file in the output folder
            output_file_path = os.path.join(output_folder, f'perturbed_{filename}')
            perturbed_df.to_csv(output_file_path, index=False)
            print(f"Perturbed dataset saved to {output_file_path}")
            print(f"Number of perturbed statements: {len(perturbed_df)}")

# Specify input and output folder paths
input_folder = 'datasets'
output_folder = 'perturbed_datasets'

# Create perturbed datasets
create_perturbed_datasets(input_folder, output_folder)