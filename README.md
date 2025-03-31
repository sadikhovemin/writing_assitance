### Formality Detection

## Installation
Run below steps to create environment and install required libraries:

```
git clone https://github.com/sadikhovemin/writing_assitance.git
cd writing_assistance
conda create --name writing-assistance python=3.10
conda activate writing-assistance
pip install -r requirements.txt
```

## Datasets
I'm using real dataset `pavlick-formality-scores` from `huggingface`. It is popular dataset used for formality detection. It contains over 10k sentences with their annotations. Annotations were created by humans. The data was collected over four genres (news, blogs, email and QA). Below is sample example of the dataset:

| domain  | avg_score | sentence                                                                                   |
|---------|-----------|--------------------------------------------------------------------------------------------|
| news    | -0.6      | Tang was employed at private-equity firm Friedman Fleischer & Lowe.                        |
| news    | 1.0       | San Francisco Mayor Gavin Newsom's withdrawal from the governor's race followed...         |
| answers | -2.8      | lol nothing worrying about that.                                                           |
| news    | 0         | She told Price she wanted to join the Police Explorers, a Boy Scouts group...              |
| news    | 1.8       | The prime minister is keen to use the autumn pre-budget statement...                       |

The `avg_score` represents the score for the sentence ranging from -3 (informal) to 3 (formal). Each sentence was annotated by multiple humans and their scores are averaged to get the final formality score `avg_score` for the sentence.

Run below to install and save the dataset
```
python scripts/preprocess_data.py
```


I'm also using synthetic dataset. Instead of picking open source relevant datasets, I tried to experiment and generate the synthetic dataset. I'm using `TinyLlama-1.1B-Chat-v1.0` model to generate synthetic dataset. It uses original real dataset `pavlick-formality-scores` which we downloaded earlier and creates sentence in opposite formality (e.g., if the sentence in the original dataset is in formal tone, it will rewrite it to informal).

Run below to create synthetic dataset:
```
python scripts/generate_synthetic.py
```

All the data related files will be preprocessed and saved to `data` folder.

## Models
I used open source model `s-nlp/roberta-base-formality-ranker` from `s-nlp` which was fine-tuned for the formality detection task. 

Reference: https://huggingface.co/s-nlp


## Evaluation

I'm evaluating results with various models trained/fine-tuned for formality detection task.

### Metrics
For evaluation I used below metrics to evalute the per-class predictions

Get per-class precision, recall, and F1 (assuming binary: 0=informal, 1=formal)

#### Accuracy
Used to measure the accuracy, how many predictions were correct in total.

#### Precision
Used to measure the precision of the model, how much of the correct predictions were actually correct / how much of the positives were actually positive.

#### Recall
Used to measure how much of total correct answers the model were able to cover. 

#### F1 Score
Harmonic mean of precision and recall. In reality, it is impossible to obtain both 1 precision and 1 recall. Thus, the F1 score provides a single number which shows how well the model has precision and recall. 

Run below to evaluate the models:
```
python scripts/evaluate_model.py --data {path_to_csv_file} --device {device}
```
Sample command:
```
python scripts/evaluate_model.py --data data/test.csv --device mps
python scripts/evaluate_model.py --data data/synthetic.csv --device mps
```



Sample result:

```json
{
        "accuracy": 0.8085,
        "precision_informal": 0.8614457831325302,
        "recall_informal": 0.7273652085452695,
        "f1_informal": 0.7887479316050745,
        "precision_formal": 0.770940170940171,
        "recall_formal": 0.8869223205506391,
        "f1_formal": 0.8248742569730224,
        "confusion_matrix": [
            [
                715,
                268
            ],
            [
                115,
                902
            ]
        ],
        "model": "RoBERTa-base (EN)"
    }
```