import logging
import numpy as np
from pprint import pprint

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.search import grid_search
from wrench import labelmodel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


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
    dataset_home = '../datasets'
    data = 'combined' #DOUBLE CHECK
    train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False,
                                                     dataset_type="TextDataset")

    print(f"Snorkel {data}")

    #### Specify the hyper-parameter search space for grid search
    search_space = {
        'Snorkel': {
            'lr': np.logspace(-6, -1, num=6, base=10),
            'l2': np.logspace(-6, -1, num=6, base=10),
            'n_epochs': [5, 10, 50, 100, 200, 300, 500, 1000],
            'seed': [123],
        }
    }

    #### Initialize label model
    label_model_name = 'Snorkel'
    label_model = getattr(labelmodel, label_model_name)

    #### Search best hyper-parameters using validation set in parallel
    n_trials = 10000
    n_repeats = 1
    target = 'acc'
    searched_paras = grid_search(label_model(), dataset_train=train_data, dataset_valid=valid_data,
                                 metric=target, direction='auto',
                                 search_space=search_space[label_model_name],
                                 tie_break_policy="random", n_repeats=n_repeats,
                                 n_trials=n_trials, parallel=True)

    model = label_model(tie_break_policy="random", **searched_paras)
    history = model.fit(dataset_train=train_data, dataset_valid=valid_data)

    # TRAIN
    #print("TRAIN:")
    #evaluate(train_data)

    # DEV
    print("DEV:")
    evaluate(valid_data)

    # TEST
    print("TEST:")
    evaluate(test_data)
