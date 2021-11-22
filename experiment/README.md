# Guide to Run Your Experiment

## File structure (example)

```bash
.
├── out                                         # Output of experiments
│   └── reproduce                               # Output directory from one experiment, named after the exp_name 
│       ├── 11-21_16-40                         # Tensorboard record
│       │   └── events.out.tfevents.1637530848.ip-172-31-24-33.46103.0
│       ├── 11-21_16-40_test_prediction.tsv     # Predicted probabilities for the test_file with n_samples * n_labels entries
│       ├── 11-21_16-47.pt                      # Checkpoint of each epoch from estimator.save()
│       ├── 11-21_16-54.pt
│       ├── 11-21_17-00.pt
│       ├── 11-21_17-07.pt
│       ├── reproduce.log                       # log file
│       └── results.json                        # Metrics obtained from calculate_metrics.py
├── README.md
├── classifier.py                               # The main script for finetuning and prediction
├── data.py                                     # Preprocessing process and dataset
├── models.py                                   # Classification models
└── utils.py                                    # Helper functions and "base" classes
```


## Implementation

To explore new model architectures or finetuning objectives, please refer to the signatures below and add your implementation in `models.py`. You can hover over the function name (in VSCode) or refer to `utils.py` for more detailed documentation. 

The following documentations may come in handy: 
* [BertModel](https://huggingface.co/transformers/model_doc/bert.html#bertmodel)
* [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer)

```python
class MyHead(nn.Module): 
    def __init__(self, *args, **kwargs): 
        raise NotImplementedError

    def forward(self, *args, **kwargs): 
        raise NotImplementedError


class MyModel(BaseClassifier): 
    def __init__(self, *args, **kwargs): 
        backbone = BertModel.from_pretrained('bert-base-cased')
        head = MyHead(*args, **kwargs)
        super().__init__(backbone, head)

    # you can redefine the forward method, or inherit BaseClassifier's by default
    # def forward(self, *args, **kwargs): 
    #     raise NotImplementedError


class MyEstimator(BaseEstimator): 
    def step(self, data): 
        raise NotImplementedError
```

Before jumping to the next step, remember to update the corresponding model and estimator in `classifier.py`. 


## Finetuning

By default, the scripts below will run finetuning on `train.tsv`, compute metrics on `dev.tsv` and make predictions on `test.tsv`. Please choose a **descriptive** and **distinguishable** name `<exp_name>` for each run of experiment to help you organize results. 

```bash
conda activate exp
python classifier.py <exp_name>
```

The default hyperparameters are set to the same as the original paper. However, you can use optional flags or hardcode your ideas. To inspect all possible options, use the command below. 

```bash
python classifier.py --help
```

To inspect the loss curve and other performance metrics, the `tensorboard` output is available at your experiment's output directory. Run the command below and view results on your browser: `localhost:<new_port>`. 

```bash
tensorboard --logdir <record_dir> --port <new_port>
```


## Evaluating test dataset

The performance on the `test.tsv` will **not** be revealed during finetuning, as it is recommended to search hyperparameters according to the performance on `dev.tsv`, which is available in the `tensorboard`'s log. 

When your are satisfied with the result, obtain the performance on the test dataset using the original authors' script: `calculate_metrics.py`. 

```bash
python calculate_metrics.py \
    --test_data <path/to/test.tsv> \
    --predictions <path/to/prediction.tsv> \
    --output <path/to/output.json> \
    --noadd_neutral
```
