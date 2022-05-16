import logging
import pickle

from wrench.dataset import load_dataset
from wrench.endmodel import EndClassifierModel
from pprint import pprint
from wrench.logging import LoggingHandler

#### Just some code to print debug information to stdout


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def predict_write(model_name, model_path):
    model = EndClassifierModel()

    model.load(model_path)

    logging.info(f"Finished loading in the EndClassifierModel from {model_path}")

    acc = model.test(test_data, 'acc', tie_break_policy="random")
    logging.info(f"Test acc is {acc}")

    predictions = model.predict(test_data, tie_break_policy="random")
    logging.info(predictions)

    with open(dataset_path + data + f"/predictions/{model_name}-predictions.pkl", "wb") as \
            open_file:
        pickle.dump(predictions, open_file)

    logging.info(f"Saved predictions {model_name}-predictions.pkl\n")

if __name__ == '__main__':

    # NB! Valida õiged mudelite indeksid!

    #### Load dataset
    dataset_path = '../datasets/'
    data = 'combined'
    bert_model_name = 'tartuNLP/EstBERT'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        # We use a pre-trained BERT model to extract features for textual classification
        # datasets. For text classification dataset, we use the outputting embedding of the [CLS] token as data feature
        extract_feature=True,
        extract_fn='bert',  # extract bert embedding
        model_name=bert_model_name,
        cache_name='combined_bert_estbert',
        dataset_type="TextDataset"
    )

    #print(train_data.examples)

    logging.info("Loaded in the combined dataset.")

    #COMBINED DATASET - consists of valence dataset test set (we have gold labels)

    #WITH .pkl
    predict_write("supervised-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3"
                  "/best_bert_supervised_valency-3_bert_estbert_model_2.pkl")
    predict_write("bert-combined", "/gpfs/space/home/citius/thesis/wrench/datasets/combined"
                           "/best_bert_combined_bert_estbert_model_0.pkl")
    predict_write("cosine-combined", "/gpfs/space/home/citius/thesis/wrench/datasets/combined"
                             "/best_cosine_combined_cosine_estbert_model_1.pkl")
    predict_write("weasel-combined", "/gpfs/space/home/citius/thesis/wrench/datasets/combined"
                             "/best_weasel_combined_weasel_estbert_model_2.pkl")
    predict_write("bert-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3/best_bert_valency-3_bert_estbert_model_3.pkl")
    predict_write("cosine-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3/best_cosine_valency"
                  "-3_cosine_estbert_model_4.pkl")
    predict_write("weasel-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3/best_weasel_valency-3_weasel_estbert_model_4.pkl")

    logging.info("Finished saving predictions for combined dataset.")

    #POSTIMEES - test
    #### Load dataset
    dataset_path = '../datasets/'
    data = 'postimees'
    bert_model_name = 'tartuNLP/EstBERT'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        # We use a pre-trained BERT model to extract features for textual classification
        # datasets. For text classification dataset, we use the outputting embedding of the [CLS] token as data feature
        extract_feature=True,
        extract_fn='bert',  # extract bert embedding
        model_name=bert_model_name,
        cache_name='postimees_bert_estbert',
        dataset_type="TextDataset"
    )

    logging.info("Loaded in the Postimees dataset.")

    #POSTIMEES DATASET TEST SET (consists of Postimees test which we have no gold labels)

    # WITH .pkl
    predict_write("supervised-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3"
                  "/best_bert_supervised_valency-3_bert_estbert_model_2.pkl")
    predict_write("bert-combined", "/gpfs/space/home/citius/thesis/wrench/datasets/combined"
                                   "/best_bert_combined_bert_estbert_model_0.pkl")
    predict_write("cosine-combined", "/gpfs/space/home/citius/thesis/wrench/datasets/combined"
                                     "/best_cosine_combined_cosine_estbert_model_1.pkl")
    predict_write("weasel-combined", "/gpfs/space/home/citius/thesis/wrench/datasets/combined"
                                     "/best_weasel_combined_weasel_estbert_model_2.pkl")
    predict_write("bert-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3/best_bert_valency-3_bert_estbert_model_3.pkl")
    predict_write("cosine-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3/best_cosine_valency"
                  "-3_cosine_estbert_model_4.pkl")
    predict_write("weasel-valency",
                  "/gpfs/space/home/citius/thesis/wrench/datasets/valency-3/best_weasel_valency-3_weasel_estbert_model_4.pkl")

    logging.info("Finished saving predictions.")
