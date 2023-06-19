import json
from pipeline import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from random import randint
import pickle

def check_predictions(args_predict):

    dataset_dir = Path(args_predict["input_dir"]) / args_predict["dataset_name"]
    dataset = DropletDataset(dataset_dir, transform=None)

    with open(args_predict["input_dir"]+args_predict["dataset_name"]+"/"+args_predict["pred_name"]+'.json') as json_file:
        predfile = json.load(json_file)
    predictions = predfile["predictions"]
    
    print("The data set contains "+str(len(dataset))+" samples")

    labels = []
    preds = []
    for img_idx in range(len(dataset.file_sizes)):
        for drop_idx in range(dataset.file_sizes[img_idx]):
            droplet, label = dataset[(img_idx,drop_idx)]
            pred = list(filter(lambda x:x["image_index"]==img_idx and x["droplet_index"]==drop_idx,predictions))[0]["prediction"]

            labels.append(label)
            preds.append(pred)

    if args_predict['predict_dead']:
        # binary classification
        labels = [int(l == 1) for l in labels]

    matrix = confusion_matrix(y_true = labels, y_pred = preds)

    confmat = {'matrix': matrix, 'y':'labels','x':'predictions','sample_size':len(preds),'dataset_name':args_predict['dataset_name']}
    with open(args_predict["model_dir"]+"confusion.pickle", "wb") as outfile:
        pickle.dump(confmat, outfile)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Labels', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    return None



def view_sample_predictions(n, args_predict):
    dataset_dir = Path(args_predict["input_dir"]) / args_predict["dataset_name"]
    dataset = DropletDataset(dataset_dir, transform=None)

    with open(args_predict["input_dir"]+args_predict["dataset_name"]+"/"+args_predict["pred_name"]+'.json') as json_file:
        predfile = json.load(json_file)
    predictions = predfile["predictions"]
    
    print("The data set contains "+str(len(dataset))+" samples")

    # Plotting random Droplets
    file_sizes = dataset.file_sizes
    m = 4
    fig, axs = plt.subplots(m, 2 * n)
    for i in range(m):
        for j in range(2 * n):
            rand_file = randint(0, len(file_sizes) - 1)
            rand_drop = randint(0, file_sizes[rand_file] - 1)
            drop, _ = dataset[(rand_file, rand_drop)]
            drop = drop.numpy().astype(dtype=float)
            pred = list(filter(lambda x:x["image_index"]==rand_file and x["droplet_index"]==rand_drop,predictions))[0]["prediction"]
            # drop_c = drop_c.numpy().astype(dtype=float)
            axs[i, j].imshow(drop, cmap="gray")
            axs[i, j].set_title(f"{pred}")
            # axs[i, 2 * j + 1].imshow(drop_c, cmap="gray")
    plt.show()
    return None