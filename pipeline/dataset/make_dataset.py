from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from random import randint
from typeguard import check_argument_types

from pipeline import ImageBatch, ImageSequence, ImageIntensityNormalization, ImageChannelExtraction
from pipeline import MetaExtractor, FilterSequence, ImageFilterDayZero, ImageFilterManual
from pipeline import DropletDataset, DropletSequence, DropletFG, DropletBiasRemoval
from pipeline import ValidatorSequence, NonEmptyValidator, AnythingGoesValidator
from pipeline import DropletOutlierDetectorSequence, DropletOutlierIsolationForest, DropletOutlierLocalFactor


filters_dict = {
    "FilterManual": ImageFilterManual,
    "FilterDayZero": ImageFilterDayZero,
}

validators_dict = {
    "AnythingGoes": AnythingGoesValidator,
    "NonEmpty": NonEmptyValidator,
}

detectors_dict = {
    "IsolationForest": DropletOutlierIsolationForest,
    "LocalFactor": DropletOutlierLocalFactor
}
    


def make_dataset(
    dataset_dir : Union[Path, str],
    image_dir : Union[Path, str],
    scoring_dir : Optional[Union[Path, str]],
    temp_dir : Union[Path, str],
    hist_dir : Optional[Union[Path, str]] = None,
    overwrite : bool = False, 
    channels : Optional[List[str]] = None,
    fg_threshold : float = 0.15,
    fg_theta : int = 100,
    fg_r : int = 1000,
    filters : List[str] = [],
    validators : List[str] = ["NonEmpty"], 
    detectors : List[str] = ["IsolationForest", "LocalFactor"],
    extract_labels : bool = True
):

    # ------------------------------------------------------------------------------------------------------------------
    # -                                             Validate arguments                                                 -
    # ------------------------------------------------------------------------------------------------------------------

    # Verify that all types are correct
    check_argument_types()

    # If unspecified, then hist_dir is set to be equal to dataset_dir
    if hist_dir is None:
        hist_dir = dataset_dir

    # Check that all ranges are valid
    dataset_dir = Path(dataset_dir)
    image_dir = Path(image_dir)
    scoring_dir = Path(scoring_dir)
    temp_dir = Path(temp_dir)
    hist_dir = Path(hist_dir)

    # Check that paths that should exists does
    for name, arg in ([("image_dir", image_dir), ("scoring_dir", scoring_dir)] if extract_labels else [("image_dir", image_dir)]):
        if not (arg.exists() and arg.is_dir()):
            raise ValueError(f"range of {name} is invalid. The path {arg} should exist and be a directory")
    
    if not ( 0 <= fg_threshold <= 1):
        raise ValueError(f"range of fg_threshold is invalid. Should be in [0, 1]")
    
    for name, arg in [("fg_theta", fg_theta), ("fg_r", fg_r)]:
        if not 0 <= arg:
            raise ValueError(f"range of {name} is invalid. Should be in [0, inf)")

    # Check that mappings are valid
    for name, arg, mapping in zip(["filters", "validators", "detectors"], [filters, validators, detectors], [filters_dict, validators_dict, detectors_dict]):
        if not all(x in mapping for x in arg):
            raise ValueError(f"range of {name} is invalid. Allowed valued are: " + ", ".join(mapping.keys()))

    # ------------------------------------------------------------------------------------------------------------------
    # -                                             Setup                                                              -
    # ------------------------------------------------------------------------------------------------------------------

    # Terminate if dataset exists and not overwriting specified
    if dataset_dir.exists() and not overwrite:
        print(f"Dataset exists at {str(dataset_dir)}. To overwrite use overwrite=True")
        return

    # Calculating META Data
    meta_extractor = MetaExtractor(out_dir=temp_dir)
    
    # Mandatory preprocessing of images
    channel_extractor = ImageChannelExtraction(
            channels=channels,
            out_dir=temp_dir
    )
    intensity_normalizer = ImageIntensityNormalization(
            out_dir=temp_dir,
            hist_dir=hist_dir,
            min_intensity=0.,
            max_intensity=1.
    )
    image_preprocessing = ImageSequence(
        channel_extractor,
        intensity_normalizer,
        out_dir=temp_dir
    )

    # Mandatory preprocessing of droplets
    droplet_transform = DropletSequence(
        DropletBiasRemoval(min_intensity=0., max_intensity=1.),
        DropletFG(tresh=fg_threshold, linspace={"theta": fg_theta, "r": fg_r} ),
    )

    # Load batch
    print("Loading image batch.", flush=True)
    batch = ImageBatch.from_dir(image_dir)
    print(f"Total number of images: {len(batch)}", flush=True)

    # Calculate META Data
    print("Calculating META Information", flush=True)
    meta_extractor.extract_batch(batch)

    # Filter Batch
    print("Filtering Images", flush=True)
    if filters is not None:
        batch = FilterSequence(*[filters_dict[f]() for f in filters])(batch)
    print(f"Total number of images after filtering: {len(batch)}", flush=True)

    # Preprocess batch
    print("Preprocessing image batch.", flush=True)
    batch = image_preprocessing(batch)

    # Make dataset from batch
    print("Turning Images into Droplets.", flush=True)

    dataset = DropletDataset.from_image_batch(
        batch=batch,
        scoring_dir=scoring_dir,
        data_dir=dataset_dir,
        extract_labels=extract_labels,
        validator=ValidatorSequence(*[validators_dict[f]() for f in validators]),
        pre_transform=droplet_transform,
        outlier_detector=DropletOutlierDetectorSequence(*[detectors_dict[f]() for f in detectors]),
        transform=None,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # -                                             Add Meta Data                                                      -
    # ------------------------------------------------------------------------------------------------------------------

    # Add pre-processing data
    additional_meta={
        "hist_dir": str(hist_dir),
        "channels": channels,
        "fg-threshold": fg_threshold,
        "fg-theta": fg_theta,
        "fg-r": fg_r,
        "filters": filters,
        "validators": validators,
        "detectors": detectors,
        "no-labels": not extract_labels,
    }
    dataset.add_metadata(additional_meta)


def view_samples(n, dataset_dir):
    dataset = DropletDataset(dataset_dir, transform=None)
    
    print("The data set contains "+str(len(dataset))+" samples")

    # Plotting random Droplets
    file_sizes = dataset.file_sizes
    m = 4
    fig, axs = plt.subplots(m, 2 * n)
    for i in range(m):
        for j in range(2 * n):
            rand_file = randint(0, len(file_sizes) - 1)
            rand_drop = randint(0, file_sizes[rand_file] - 1)
            drop, y = dataset[(rand_file, rand_drop)]
            drop = drop.numpy().astype(dtype=float)
            # drop_c = drop_c.numpy().astype(dtype=float)
            axs[i, j].imshow(drop, cmap="gray")
            axs[i, j].set_title(f"{y}")
            # axs[i, 2 * j + 1].imshow(drop_c, cmap="gray")
    plt.show()
    return None


if __name__ == "__main__":
    raise RuntimeError('not meant as main script')