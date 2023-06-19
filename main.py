import argparse
from pathlib import Path
import os

from pipeline import * 
from pipeline.training.train import train, image_transforms_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, help="The directory for the dataset.", metavar="", required=True)
    parser.add_argument("--model_dir", type=Path, default=Path("models"), help="The directory for the dataset.", metavar="")
    parser.add_argument("--no-prefix", action="store_false", dest="use_prefix", help="Use plain paths. If not activated, automatically detect if on Euler and use the scratic directory.")
    parser.set_defaults(use_prefix=True)
    parser.add_argument("--model-name", type=str, help="Name of the trained model being created", metavar="", required=True)
    parser.add_argument("--freeze", action="store_true", dest="freeze", help="Freeze parts of the pretrained model.")
    parser.set_defaults(freeze=False)
    parser.add_argument("--pretrained-name", type=str, default="timm-resnet34", help="Name of the pretrained trained model being loaded as a startpoint", metavar="")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size during training and validation", metavar="")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs", metavar="")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Batch size during training and validation", metavar="")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Fraction of samples used for validation", metavar="")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for reproducability", metavar="")
    parser.add_argument("--validate-every", type=int, default=150, help="How often to validate", metavar="")
    parser.add_argument("--predict-dead", action="store_true", dest="predict_dead", help="Only predict binary if cell is dead.")
    parser.set_defaults(predict_dead=False)
    parser.add_argument("--wb", action="store_true", dest="wb", help="Log using Weights & Biases.")
    parser.set_defaults(wb=False)
    parser.add_argument("--image-transforms", type=str, default=["RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "ColorJitter"], nargs="+", choices=image_transforms_dict.keys(), help="Transforms to apply to the droplet images", metavar="")

    args = parser.parse_args()

    # Get if on Euler
    cwd = os.getcwd()
    if "cluster" in cwd:
        uname = cwd.split("/")[3]
        euler_prefix = Path(f"/cluster/scratch/{uname}/")
    else:
        euler_prefix = Path("")

    # Update pats as requested
    if args.use_prefix:
        args.dataset_dir = euler_prefix.joinpath(args.dataset_dir)

    # Call make_dataset with given arguments
    # use_prefix is only used in this specific script
    no_keep = ["use_prefix"]
    train(**{key: value for key, value in args._get_kwargs() if not key in no_keep})

    
