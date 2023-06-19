import datetime
import json 
from pathlib import Path
from typeguard import check_argument_types
from typing import Union

import torch.nn as nn
from torch.optim import Adam 
from torchmetrics import Accuracy	
from torchmetrics.classification import MulticlassF1Score
from torchvision.transforms import *

from pipeline import * 
from pipeline.training.train_utils import collate_fn
from pipeline import Trainer, Metric 


image_transforms_dict = {
    "RandomHorizontalFlip": RandomHorizontalFlip(),
    "RandomVerticalFlip": RandomVerticalFlip(),
    "RandomRotation": RandomRotation(90),
    "ColorJitter": ColorJitter(),
    "RandomInvert": RandomInvert()
}


def save_args(args, target):
    if not target.exists(): target.mkdir(parents=True, exist_ok=True)
    with open(target/"args.json", "w") as outfile:
        json.dump(args, outfile)


def train(
    dataset_dir : Union[Path, str],
    model_dir : Union[Path, str],
    model_name : str,
    pretrained_name : str,
    freeze : bool = False,
    batch_size : int = 50,
    epochs : int = 10,
    learning_rate : float = 1e-4,
    validation_split : float = 0.05,
    seed : int = 42,
    validate_every : int = 150,
    predict_dead : int = True,
    wb : bool = False,
    image_transforms : List[str] = ["RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "ColorJitter"]
):

    # ------------------------------------------------------------------------------------------------------------------
    # -                                             Validate arguments                                                 -
    # ------------------------------------------------------------------------------------------------------------------
    

    # Verify that all types are correct
    check_argument_types()

    # Check that all ranges are valid
    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)
    
    if not ( 0 <= validation_split <= 1):
        raise ValueError(f"range of validation_split is invalid. Should be in [0, 1]")
    
    for name, arg in [("batch_size", batch_size), ("epochs", epochs), ("learning_rate", learning_rate), ("seed", seed)]:
        if not 0 <= arg:
            raise ValueError(f"range of {name} is invalid. Should be in [0, inf)")

    # Check that mappings are valid
    for name, arg, mapping in zip(["image_transforms"], [image_transforms], [image_transforms]):
        if not all(x in mapping for x in arg):
            raise ValueError(f"range of {name} is invalid. Allowed valued are: " + ", ".join(mapping.keys()))

    # ------------------------------------------------------------------------------------------------------------------
    # -                                             Setup                                                              -
    # ------------------------------------------------------------------------------------------------------------------

    experiment_name = str(model_name)+ f"_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up transforms

    # Use specified image transforms
    transform_img = Compose([image_transforms_dict[t] for t in image_transforms])

    # If only predicing dead labels, map labels to binary vector indicating if the label is 1
    if predict_dead:
        transform_label = lambda l: (l == 1).to(dtype=l.dtype, device=l.device)
        num_classes = 2
        class_weights = [1.0, 1.0]
        
    else:
        transform_label = lambda l: l
        num_classes = 4
        class_weights = [1/0.3,1/0.5,1/0.15,1/0.15]

    # Load data 
    datamodule = DropletDataModule(
        data_dir=dataset_dir, 
        batch_size=batch_size, 
        tts=validation_split, 
        collate_fn=lambda batch: collate_fn(batch, transform_img=transform_img, transform_labels=transform_label), 
        seed=seed
    )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Model Architecture
    cnn = load_pretrained(
        name=model_name, 
        model_name=pretrained_name, 
        in_chans=1, # normally 3 for RGB-images
        freeze_extractor=freeze
    )
    output_head = OutputHead(n_in=cnn.get_out_dim(), n_out=num_classes)

    # Model definition
    model = nn.Sequential(
        cnn,
        output_head,
    ).to(device)

    # Define metrics
    weights = torch.tensor(class_weights).to(device=device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    metrics = [
      Metric(name, method) for name, method in {
          "accuracy": Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device=device),
          "f1score": MulticlassF1Score(num_classes=num_classes).to(device=device)
      }.items()
    ]

    # Save arguments
    save_args({
        "pretrained_name" : pretrained_name,
        "freeze" : freeze,
        "batch_size" : batch_size,
        "epochs" : epochs,
        "learning_rate" : learning_rate,
        "validation_split" : validation_split,
        "seed" : seed,
        "validate_every" : validate_every,
        "predict_dead" : predict_dead,
        "image_transforms" : image_transforms
    }, model_dir / experiment_name)

    # Configs for Weights and Biases
    transform_helper = nn.Sequential(nn.Identity())
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "transforms1": transform_helper,
        "transforms2": transform_helper,
        "device": device,
        "model": model,
    }

    # ------------------------------------------------------------------------------------------------------------------
    # -                                             Run training                                                       -
    # ------------------------------------------------------------------------------------------------------------------

    # Create trainer
    trainer = Trainer(
        model=model, 
        data_pipeline_train=transform_helper, 
        data_pipeline_val=transform_helper, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device, 
        metrics=metrics, 
        validate_every=validate_every, 
        use_wandb=wb, 
        wandb_args={
            "columns": ["Predicted", "Expected"], 
            "project_name": "dsl_mini_tumors", 
            "experiment_name": experiment_name, 
            "config": config, 
        },
        directory=f"{str(model_dir / experiment_name)}",
        name=model_name
    )
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=epochs)

if __name__ == '__main__':
    raise RuntimeError('not meant as main script')
