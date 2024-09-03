import io
import hydra
import torch
from hydra.core.config_store import ConfigStore
from data_utils import *
from modeling import *
from train_utils import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from torchviz import make_dot

import os

# Check the current working directory
print("Current working directory:", os.getcwd())
# Use this function to load the weights
def define_and_load_model(conf):
    # Define the model
    model = Model_Recursive_LSTM_v2(
        input_size=conf.model.input_size,
        comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
        drops=list(conf.model.drops),
        loops_tensor_size=8,
        device=conf.testing.gpu,
    )
    # Load the trained model weights
    model.load_state_dict(
        torch.load(
            conf.testing.testing_model_weights_path,
            map_location=conf.testing.gpu,
        )
    )
    model = model.to(conf.testing.gpu)
    
    # Set the model to evaluation mode
    model.eval()
    return model


def evaluate(conf, model):
    
    print("Loading the dataset...")
    val_ds, val_bl, val_indices, _ = load_pickled_repr(
        os.path.join(conf.experiment.base_path ,'pickled/pickled_')+Path(conf.data_generation.valid_dataset_file).parts[-1][:-4], 
        max_batch_size = 128, 
        store_device=conf.testing.gpu, 
        train_device=conf.testing.gpu
    )
#     print(val_bl)
#     print(val_bl.shape())
    print("Evaluation...")
    val_df = get_results_df(val_ds, val_bl, val_indices, model, train_device = conf.testing.gpu)


    

    predictions = torch.tensor(val_df["prediction"])  # Example predictions
    labels = torch.tensor(val_df["target"])  # Example labels
    print(predictions)
    dot = make_dot(predictions.mean(), params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('torchviz-sample')
    predictions = predictions.numpy()
    labels = labels.numpy()
    
    for i in range (len(predictions)):
        if(predictions[i]> 0.5):
            predictions[i]=1.0
            
        else:
            predictions[i]=0.0
            

    
    y_test = labels
    y_pred= predictions
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    color = 'white'
#     matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
#     matrix.ax_.set_title('Confusion Matrix', color=color)
#     plt.xlabel('Predicted Label', color=color)
#     plt.ylabel('True Label', color=color)
#     plt.gcf().axes[0].tick_params(colors=color)
#     plt.gcf().axes[1].tick_params(colors=color)


    
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print("precision: ", precision)
    print("recall: ", recall)
    
    return f1
    

#     val_scores = get_scores(val_df)
#     return dict(
#         zip(
#             ["nDCG", "nDCG@5", "nDCG@1", "Spearman_ranking_correlation", "MAPE"],
#             [item for item in val_scores.describe().iloc[1, 1:6].to_numpy()],
#         )
#     )


@hydra.main(config_path="conf", config_name="config")
def main(conf):
    print("Defining and loading the model using parameters from the config file")
    model = define_and_load_model(conf)
    print(f"Validating on the dataset: {conf.data_generation.valid_dataset_file}")
    scores = evaluate(conf, model)
    print(f"Evaluation scores are:\n{scores}")

if __name__ == "__main__":
    main()
