import logging
import torch

from wrench import labelmodel
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.labelmodel import Snorkel
from wrench.endmodel import Cosine

#### Just some code to print debug information to stdout
from wrench.search import grid_search

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

if __name__ == '__main__':
    #### Load dataset
    dataset_path = '../datasets/'
    data = 'valency-3'
    bert_model_name = 'tartuNLP/EstBERT'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        # We use a pre-trained BERT model to extract features for textual classification
        # datasets. For text classification dataset, we use the outputting embedding of the [CLS] token as data feature
        extract_feature=True,
        extract_fn='bert',  # extract bert embedding
        model_name=bert_model_name,
        cache_name='estbert_cosine',
        dataset_type="TextDataset"
    )

    # Best value: 0.65625, Best paras: {'lr': 0.1, 'l2': 0.01, 'n_epochs': 50, 'seed': 123}
    #### Run label model: Snorkel
    label_model = Snorkel(
        lr=0.1,
        l2=0.01,
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
    aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data)

    #### Search Space
    search_space = {
        "batch_size": [32],
        "optimizer_lr": [1e-6, 1e-5],
        "optimizer_weight_decay": [1e-4],
        "teacher_update": [50, 100, 200],
        "thresh": [0.2, 0.4, 0.6, 0.8],
        "lamda": [0.01, 0.05, 0.1],
        "mu": [1],
        "dropout": [0.1],
    }

    # Trial 213 finished with value: 0.7017045454545454 and parameters:
    # {'batch_size': 16, 'optimizer_lr': 5e-05, 'optimizer_weight_decay': 0.0001, 'teacher_update': 800,
    # 'thresh': 0.4, 'lamda': 0.01, 'mu': 0.1, 'margin': 1, 'dropout': 0.3}. Best is trial 213 with value: 0.7017045454545454.

    model = Cosine(
        batch_size=16,
        teacher_update=800,
        margin=1.0,
        thresh=0.4,  # xi - confidence threshold
        mu=0.1,  # mu - weight for contrastive regularization
        lamda=0.01,  # lambda - weight for confidence regularization
        dropout=0.3,
        optimizer_lr=5e-5,
        optimizer_weight_decay=1e-4,
        backbone='BERT',
        backbone_model_name=bert_model_name,

        # For gradient-based optimization, we adopt AdamW Optimizer and linear learning rate
        # scheduler;
        optimizer='AdamW',
        use_lr_scheduler=True,
        lr_scheduler='default',
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
    )

    model.fit(dataset_train=train_data,
              dataset_valid=valid_data,
              y_train=aggregated_soft_labels,
              soft_labels=True,
              evaluation_step=10,
              metric='acc',
              #patience=50, #and we early stop the training processbased on the evaluation metric
              # values on validation set
              device=device)

    model.save(dataset_path + data + "/best_cosine_model.pkl")

    acc = model.test(valid_data, 'acc', tie_break_policy="random")

    logger.info(model.predict(valid_data))

    logger.info(f'end model (COSINE) VALID acc: {acc}')

    acc = model.test(test_data, 'acc', tie_break_policy="random")

    logger.info(model.predict(test_data))

    logger.info(f'end model (COSINE) TEST acc: {acc}')
