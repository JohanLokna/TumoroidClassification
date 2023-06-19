import argparse
from pathlib import Path

from pipeline.dataset.make_dataset import make_dataset, filters_dict, validators_dict, detectors_dict

# OMP error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, help="The directory for the dataset.", metavar="", required=True)
    parser.add_argument("--image_dir", type=Path, help="The input directory for the image files", metavar="", required=True)
    parser.add_argument("--scoring_dir", type=Path, help="The input directory for the scoring files", metavar="", required=True)
    parser.add_argument("--temp_dir", type=Path, help="The base directory for temporal stroage during the pre-processing.", metavar="", required=True)
    parser.add_argument("--hist_dir", type=Path, help="The directory for loading pre-computed statistics. If not specified, then set to same values as dataset_dir.", metavar="")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any pre-existing dataset in the same destination")
    parser.set_defaults(overwrite=False)
    parser.add_argument("--no-prefix", action="store_false", dest="use_prefix", help="Use plain paths. If not activated, automatically detect if on Euler and use the scratic directory.")
    parser.set_defaults(use_prefix=True)
    parser.add_argument("--channels", type=str, nargs="+", help="Channels to extract from images. If not activated, all channels are extracted.", metavar="")
    parser.add_argument("--fg-threshold", type=float, default=0.15, help="Threshold parameter for foreground extraction", metavar="")
    parser.add_argument("--fg-theta", type=int, default=100, help="Number of angular discretization points in the foreground extraction", metavar="")
    parser.add_argument("--fg-r", type=int, default=1000, help="Number of radial discretization points in the foreground extraction", metavar="")
    parser.add_argument("--filters", type=str, default=[], nargs="+", choices=filters_dict.keys(), help="Filters to apply to the batches", metavar="")
    parser.add_argument("--validators", type=str, default=["NonEmpty"], nargs="+", choices=validators_dict.keys(), help="Validators to apply to the batches", metavar="")
    parser.add_argument("--detectors", type=str, default=["IsolationForest", "LocalFactor"], nargs="+", choices=detectors_dict.keys(), help="Detectors to apply to the batches", metavar="")
    parser.add_argument("--no-labels", action="store_false", dest="extract_labels", help="No extraction of labels. Necessary for prediction on new datasets without annotations")
    parser.set_defaults(extract_labels=True)
    args = parser.parse_args()

    # If unspecified, then hist_dir is set to be equal to dataset_dir
    if args.hist_dir is None:
        args.hist_dir = args.dataset_dir


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
        args.image_dir = euler_prefix.joinpath(args.image_dir)
        args.scoring_dir = euler_prefix.joinpath(args.scoring_dir)
        args.temp_dir = euler_prefix.joinpath(args.temp_dir)
        args.hist_dir = euler_prefix.joinpath(args.hist_dir)

    # Call make_dataset with given arguments
    # use_prefix is only used in this specific script
    no_keep = ["use_prefix"]
    make_dataset(**{key: value for key, value in args._get_kwargs() if not key in no_keep})
