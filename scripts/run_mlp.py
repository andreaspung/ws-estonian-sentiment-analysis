import logging
from pprint import pprint

import torch
import numpy as np
from wrench.dataset import load_dataset
from wrench.labelmodel import Snorkel
from wrench.logging import LoggingHandler
from wrench.search import grid_search
from wrench.endmodel import EndClassifierModel

#### Just some code to print debug information to stdout
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
    cache_name = f"{data}_estbert_mlp"
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        #We use a pre-trained BERT model to extract features for textual classification
        #datasets. For text classification dataset, we use the outputting embedding of the [CLS] token as data feature
        extract_feature=True,
        extract_fn='bert',  # extract bert embedding
        model_name=bert_model_name,
        cache_name=cache_name,
        dataset_type="TextDataset"
    )

    print(f"MLP {cache_name}")
    # Best value: 0.65625, Best paras: {'lr': 0.1, 'l2': 0.001, 'n_epochs': 50, 'seed': 123}
    #### Run label model: Snorkel
    label_model = Snorkel(
        lr=0.1,
        l2=1e-03,
        n_epochs=50,
        seed=123
    )
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )

    acc = label_model.test(test_data, 'acc', tie_break_policy="random")
    logger.info(f'label model test acc: {acc}')

    #### Filter out uncovered training data
    aggregated_hard_labels = label_model.predict(train_data, tie_break_policy="random")
    aggregated_soft_labels = label_model.predict_proba(train_data, tie_break_policy="random")

    #### Search Space
    search_space = {
        "batch_size": [32, 128, 512],
        'optimizer_lr': np.logspace(-5, -1, num=5, base=10),
        'optimizer_weight_decay': np.logspace(-5, -1, num=5, base=10),
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    #### Initialize the model: MLP
    model = EndClassifierModel(
        backbone='MLP',
        # For gradient-based optimization, we adopt AdamW Optimizer and linear learning rate
        # scheduler;
        optimizer='AdamW',
        use_lr_scheduler=True,
        lr_scheduler='default',
        # #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR

    )

    #### Search best hyper-parameters using validation set in parallel
    n_trials = 100000
    n_repeats = 1
    searched_paras = grid_search(
        model,
        dataset_train=train_data,
        y_train=aggregated_soft_labels,
        dataset_valid=valid_data,
        metric='acc',
        direction='auto',
        search_space=search_space,
        n_repeats=n_repeats,
        n_trials=n_trials,
        parallel=True,
        device=device,
    )

    print("Evaluation")
    for i in range(5):
        print("i", i)
        #### Run end model: MLP
        model = EndClassifierModel(
            backbone='MLP',
            # For gradient-based optimization, we adopt AdamW Optimizer and linear learning rate
            # scheduler;
            optimizer='AdamW',
            use_lr_scheduler=True,
            lr_scheduler='default',
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
            **searched_paras
        )
        model.fit(
            dataset_train=train_data,
            y_train=aggregated_soft_labels,
            dataset_valid=valid_data,
            metric='acc',
            #patience=50, #and we early stop the training process based on the evaluation metric
            # values on validation set
            device=device
        )

        model.save(dataset_path + data + f"/best_mlp_{cache_name}_model_{i}.pkl")

        # TRAIN
        print("TRAIN:")
        evaluate(train_data)

        # DEV
        print("DEV:")
        evaluate(valid_data)

        # TEST
        print("TEST:")
        evaluate(test_data)
