# Toxic Comment Classification with NLP

An NLP project exploring different approaches to toxic comment classification, from traditional machine-learning models to transformers and large language models (LLMs).

## Project overview

Toxic comments can make online spaces less welcoming and discourage participation. This project compares several NLP approaches for identifying toxic comments and examines their predictive performance, trade-offs, and inference time.

The notebook covers the workflow from text preprocessing and feature engineering to model evaluation and visual comparison.

## Approaches

- **Word embeddings:** Word2Vec, GloVe, and fastText combined with logistic regression and linear SVM classifiers.
- **Sentence embeddings:** MiniLM embeddings used as input to traditional classifiers.
- **Transformer classification:** A pretrained DistilBERT toxicity model applied to raw comments.
- **Prompt-based classification:** FLAN-T5-base tested with zero-shot, few-shot, and role-based prompts.
- **LLM API experiment:** GPT-5.2 evaluated as an additional prompt-based approach.

## Evaluation

The project compares classification metrics such as precision, recall, and ROC-AUC where applicable, alongside inference time.

Embedding-based models and the transformer are evaluated on the test set. Since LLM inference is slower, prompt-based approaches are also evaluated on a balanced subset of 500 comments: 250 toxic and 250 non-toxic. This makes it possible to compare all approaches on the same subset.

Detailed metrics and visual comparisons are available in the notebook.

## What this project demonstrates

- Text preprocessing and NLP feature engineering
- Word-level and sentence-level embeddings
- Logistic regression and linear SVM classification
- Transformer and LLM inference
- Zero-shot and few-shot prompt engineering
- Model evaluation with classification metrics
- Comparison of predictive performance and inference time
- Visualization and interpretation of experimental results

## Repository contents

- `toxic-comment-nlp.ipynb` : notebook containing the analysis, experiments, and results
- `data.csv` : dataset used by the notebook

## Running the notebook

Open `toxic-comment-nlp.ipynb` in Jupyter or Google Colab. Make sure the dataset is available at the path expected by the notebook and install the required Python libraries.

The GPT-5.2 experiment may require API access and credentials. Never commit API keys or other secrets to the repository.
