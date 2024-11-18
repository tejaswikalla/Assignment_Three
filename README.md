The code for the supervised analysis is primarily based on Azaria and Mitchell's paper, *"The Internal State of an LLM Knows When It's Lying."* The datasets used for this work are either directly sourced from their research or derived from it.

To generate embeddings, you can use `GenerateEmbeddings.py` or `LLaMa_generate_embeddings.py` on the selected datasets. These scripts extract embeddings for the last token at specified layers of the chosen model. The embeddings are saved as CSV files, and the parameters can be configured using either a `config.json` file or command-line arguments. Ensure that labeled datasets are used for training purposes.

Next, use TrainProbes.py to train on the embeddings from the selected datasets. The script allows you to specify many parameters via the config file or command-line options. It tests the probes on a separate evaluation dataset and saves the best-performing probes based on accuracy.

To apply the training to a new dataset, run Generate_Embeddings.py on the new dataset, ensuring that the model, layer, and other parameters match those used for training. The predictions, along with the average prediction, will be saved.

The accuracy of the SAPLMA model was compared with BERT (calculated using `Bert.py`), as well as with 3-shot and 5-shot learning models (evaluated using `train_few-shot.py`).

To evaluate the robustness of the models, perturbed datasets were generated from the original ones using `perturbations.py`. These perturbations included adding fillers to sentences, introducing typos, and altering word order to test the models under varied conditions.
