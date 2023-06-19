import numpy as np
import pandas as pd
from pathlib import Path


class DropletExtractor:

    extra_padding = 80

    @staticmethod
    def extract_droplet(x_coord, y_coord, major_axis_len, img: np.ndarray, channel_names, channel_count: int):
        """Extracts droplet image from bigger image file."""

        # Extract channel names
        ignore_channels = []
        if "BF" in channel_names:                           # BF channel is needed
            bf_channel = channel_names.index("BF")
            ignore_channels.append(bf_channel)
        else:
            raise ValueError("BF Channel is missing. BF Channel is however needed for the analysis.")

        if "TRITC" in channel_names:                        # If a orange channel is present we want to ignore it
            orange_channel = channel_names.index("TRITC")
            ignore_channels.append(orange_channel)

        # Dimension of image
        x_dim = img.shape[1]
        y_dim = img.shape[2]

        # Find indices where droplet is located within image. Indices need to be integers with coordinates within image
        left_x = max(int(x_coord - major_axis_len/2 - DropletExtractor.extra_padding), 0)
        right_x = min(int(x_coord + major_axis_len/2 + DropletExtractor.extra_padding), x_dim - 1) 
        bottom_y = max(int(y_coord - major_axis_len/2 - DropletExtractor.extra_padding), 0)
        top_y = min(int(y_coord + major_axis_len/2) + DropletExtractor.extra_padding, y_dim - 1) 

        # Extract droplet from image
        img_droplet = img[bf_channel, left_x:right_x, bottom_y:top_y]

        # Get drug idx
        non_img_channels = [i for i in range(channel_count) if i not in ignore_channels]
        img_drugs = img[non_img_channels, left_x:right_x, bottom_y:top_y]
        drug_idx = np.argmax(img_drugs.reshape((len(non_img_channels), -1)).mean(axis=1))

        return img_droplet, drug_idx

    @staticmethod
    def extract_image(image,
                      scoring_path: Path,
                      extract_labels: bool = True):
        """Extracts droplets and labels from one image file and the corresponding scoring file."""
        
        # Read labels
        if extract_labels:
            df_labels = pd.read_excel(scoring_path, sheet_name=1, header=None, names=["DropIdx", "label"])
            labels = df_labels["label"].to_numpy().astype(np.int16)
        else:
            labels = []
        
        # Relevant columns
        df_info = pd.read_excel(scoring_path, sheet_name=0)
        df_info = df_info[["DropIdx", "MajorAxisLength", "TrueCentroidX", "TrueCentroidY"]]

        # Sanity check
        if extract_labels:
            if not (df_info["DropIdx"].to_numpy() == df_labels["DropIdx"].to_numpy()).all():
                raise RuntimeError(f"Bad indecies: {scoring_path}")

        channel_names = image.meta["channelNames"]              # Extract information that only depends on the
        channel_count = image.meta["channelCount"]              # image, not droplet. This prevents memory issues
        img = image.img                                         # further down the lane
        droplet_images, drugs = [], []
        for _, row in df_info.iterrows():
            
            # Extract positional data
            x_coord = row["TrueCentroidX"]
            y_coord = row["TrueCentroidY"]
            major_axis_len = row["MajorAxisLength"]

            # Find and extract droplet in image
            img_droplet, drugs_idx = DropletExtractor.extract_droplet(x_coord, y_coord, major_axis_len, img,
                                                                      channel_names, channel_count)
            droplet_images.append(img_droplet)
            drugs.append(drugs_idx)

            if not extract_labels:
                labels.append(np.nan)
        drugs = np.array(drugs)
        if not extract_labels:
            labels = np.array(labels).astype(np.int16)
        return droplet_images, labels, drugs
