import logging
from pprint import pprint

import torch

from wrench import labelmodel
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.labelmodel import MeTaL, Snorkel
from wrench.endmodel import Cosine, EndClassifierModel

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
    data = 'combined'
    bert_model_name = 'tartuNLP/EstBERT'
    cache_name = f"{data}_cosine_estbert"
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

    print(f"Cosine {cache_name}")
    # Best value: 0.65625, Best paras: {'lr': 0.1, 'l2': 0.001, 'n_epochs': 50, 'seed': 123}
    # COMBINED: Best value: 0.6392045454545454, Best paras: {'lr': 0.01, 'l2': 1e-06, 'n_epochs': 500, 'seed': 123}
    #### Run label model: Snorkel
    label_model = Snorkel(
        lr=0.01,
        l2=1e-06,
        n_epochs=500,
        seed=123
    )
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )

    acc = label_model.test(test_data, 'acc', tie_break_policy="random")
    logger.info(f'label model test acc: {acc}')

    #### Filter out uncovered training data
    #aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data)

    # COSINE [END: BEST VAL / PARAMS] Best value: 0.7073863636363636, Best paras: {'batch_size': 16, 'optimizer_lr': 5e-05, 'optimizer_weight_decay': 0.0001, 'teacher_update': 100, 'thresh': 0.5, 'lamda': 0.01, 'mu': 1, 'margin': 0.1, 'dropout': 0.4}

    searched_paras = {'batch_size': 16, 'optimizer_lr': 5e-05, 'optimizer_weight_decay': 0.0001, 'teacher_update': 100, 'thresh': 0.5, 'lamda': 0.01, 'mu': 1, 'margin': 0.1, 'dropout': 0.4}

    print("Evaluation")
    for i in range(5):
        print("i", i)
        model = Cosine(
            backbone='BERT',
            backbone_model_name=bert_model_name,

            # For gradient-based optimization, we adopt AdamW Optimizer and linear learning rate
            # scheduler;
            optimizer='AdamW',
            use_lr_scheduler=True,
            lr_scheduler='default',
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
            **searched_paras
        )

        model.fit(dataset_train=train_data,
                  dataset_valid=valid_data,
                  y_train=aggregated_soft_labels,
                  soft_labels=True,
                  evaluation_step=10,
                  metric='acc',
                  # patience=50, #and we early stop the training processbased on the evaluation metric
                  # values on validation set
                  device=device)

        model.save(dataset_path + data + f"/best_cosine_{cache_name}_model_{i}.pkl")

        # TRAIN
        #print("TRAIN:")
        #evaluate(train_data)

        # DEV
        print("DEV:")
        evaluate(valid_data)

        # TEST
        print("TEST:")
        evaluate(test_data)
