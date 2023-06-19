import matplotlib.pyplot as plt
from random import randint
from pipeline import *
from pipeline.dataset.make_dataset import make_dataset, view_samples
from .PretrainedModel import timmModel, TorchPretrainedModel
from ..training.train_utils import collate_fn
import json


def load_trained_model(args_predict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load(
            args_predict["model_dir"]+args_predict["model_name"]+".pth", map_location=device
            )
    model.eval()
    return model


def predict(args_predict):
    model = load_trained_model(args_predict)
    print("successfully loaded model")
    dataset = DropletDataset(Path(args_predict["input_dir"]+args_predict["dataset_name"]), load_drug=True)
    
    print("File sizes are", dataset.file_sizes)
    preds = []
    for img_idx in range(len(dataset.file_sizes)):
        for drop_idx in range(dataset.file_sizes[img_idx]):
            droplet, _, drug = dataset[(img_idx,drop_idx)]
            with torch.no_grad():
                prediction = int(torch.argmax(model.forward(torch.unsqueeze(torch.unsqueeze(droplet,0),0))))
            preds.append(
                {
                    "image_index": img_idx,
                    "droplet_index": drop_idx,
                    "prediction": prediction, 
                    "drug": int(drug)
                }
            )
    predfile = args_predict
    predfile.update({"predictions":preds})
    output_dir = Path(args_predict["input_dir"]) / args_predict["dataset_name"]
    if not output_dir.exists(): output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/(args_predict["pred_name"]+".json"), "w") as outfile:
        json.dump(predfile, outfile)
    print("successfully saved predictions to data directory")
    return None

    