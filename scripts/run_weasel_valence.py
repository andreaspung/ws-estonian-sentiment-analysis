import logging
from pprint import pprint

import torch

from wrench.classification import WeaSEL
from wrench.dataset import load_dataset
from wrench.labelmodel import Snorkel
from wrench.logging import LoggingHandler
#### Just some code to print debug information to stdout
from wrench.search import grid_search

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')


def evaluate(data):
    acc = model.test(data, "acc", tie_break_policy="random")
    auc = model.test(data, "auc", tie_break_policy="random")

    f1_binary = model.test(data, "f1_binary", tie_break_policy="random")
    f1_micro = model.test(data, "f1_micro", tie_break_policy="random")
    f1_macro = model.test(data, "f1_macro", tie_break_policy="random")
    f1_weighted = model.test(data, "f1_weighted", tie_break_policy="random")

    recall_binary = model.test(data, "recall_binary", tie_break_policy="random")
    recall_micro = model.test(data, "recall_micro", tie_break_policy="random")
    recall_macro = model.test(data, "recall_macro", tie_break_policy="random")
    recall_weighted = model.test(data, "recall_weighted", tie_break_policy="random")

    precision_binary = model.test(data, "precision_binary", tie_break_policy="random")
    precision_micro = model.test(data, "precision_micro", tie_break_policy="random")
    precision_macro = model.test(data, "precision_macro", tie_break_policy="random")
    precision_weighted = model.test(data, "precision_weighted", tie_break_policy="random")

    logloss = model.test(data, "logloss", tie_break_policy="random")
    brier = model.test(data, "brier", tie_break_policy="random")

    print(f"{acc} {f1_weighted} {precision_weighted} {recall_macro}")

    results = {
        'acc': acc,
        'auc': auc,
        'f1_binary': f1_binary,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'recall_binary': recall_binary,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'precision_binary': precision_binary,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'logloss': logloss,
        'brier': brier,
    }

    pprint(results)


if __name__ == '__main__':
    #### Load dataset
    dataset_path = '../datasets/'
    data = 'valency-3'
    bert_model_name = 'tartuNLP/EstBERT'
    cache_name = f"{data}_weasel_estbert"
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        # We use a pre-trained BERT model to extract features for textual classification
        # datasets. For text classification dataset, we use the outputting embedding of the [CLS] token as data feature
        extract_feature=True,
        extract_fn='bert',  # extract bert embedding
        model_name=bert_model_name,
        cache_name=cache_name,
        dataset_type="TextDataset"
    )

    print(f"Weasel fine-tuning {cache_name}")

    """
    #### Search Space
    search_space = {
        "batch_size": [16],
        "hidden_size": [100, 200, 500],
        "optimizer_lr": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        "optimizer_weight_decay": [1e-4],
        "temperature": [0.1, 0.3, 0.5, 1.0],
        "dropout": [0.3, 0.4, 0.5],
    }

    print("Search space", search_space)

    #### Run WeaSEL
    model = WeaSEL(
        backbone='BERT',
        backbone_model_name=bert_model_name,
        n_steps=10000,
        backbone_fine_tune_layers=-1,  # fine  tune all
        optimizer='AdamW',
        use_lr_scheduler=True,
        lr_scheduler='default'
    )

    #### Search best hyper-parameters using validation set in parallel
    n_trials = 100000
    n_repeats = 1
    searched_paras = grid_search(
        model,
        dataset_train=train_data,
        dataset_valid=valid_data,
        patience=500,
        metric='acc',
        direction='auto',
        search_space=search_space,
        n_repeats=n_repeats,
        n_trials=n_trials,
        parallel=False,
        device=device,
    )
    """

    searched_paras = {'batch_size': 16, 'hidden_size': 100, 'optimizer_lr': 5e-06, 'optimizer_weight_decay': 0.0001, 'temperature': 1.0, 'dropout': 0.4}

    print("Evaluation")
    for i in range(5):
        print("i", i)
        #### Run WeaSEL
        model = WeaSEL(
            n_steps=10000,
            backbone='BERT',
            backbone_model_name=bert_model_name,
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            **searched_paras
        )

        model.fit(dataset_train=train_data,
                  dataset_valid=valid_data,
                  patience=500,
                  evaluation_step=10,
                  metric='acc',
                  # patience=50, #and we early stop the training process based on the evaluation
                  # metric
                  # values on validation set
                  device=device)

        model.save(dataset_path + data + f"/best_weasel_{cache_name}_model_{i}.pkl")

        # TRAIN
        print("TRAIN:")
        evaluate(train_data)

        # DEV
        print("DEV:")
        evaluate(valid_data)

        # TEST
        print("TEST:")
        evaluate(test_data)
