import torch

from . import DropletTransform


class DropletBiasRemoval(DropletTransform):

    def __init__(self, min_intensity: float = 0, max_intensity: float = 1,
                 min_quantile: float = 0.05, max_quantile: float = 0.98) -> None:
        """
        Instead of Bias Field Removal on the whole image we deal with it on the Droplet Level.
        This is intuitively justified by the fact, that the bias field is roughly constant on one single droplet (due to their
        small size). Furthermore it is quite rare that a subplot line intersects with single droplets.
        """
        super().__init__()
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

    def transform(self, drop: torch.FloatTensor) -> torch.FloatTensor:
        """
        Min - Max Normalization to approximate constant multiplicative factor.
        """

        # Get quantiles
        l_intensity = torch.quantile(drop, self.min_quantile)
        u_intensity = torch.quantile(drop, self.max_quantile)

        # Moves the min_quantile to 0
        drop -= l_intensity

        # Truncates below l_intensity
        drop[drop < 0] = 0
        
        # Truckate above moved version of u_intensity
        m = u_intensity - l_intensity
        drop[drop > m] = m

        # Rescale and translate
        new_m = self.max_intensity - self.min_intensity
        drop = drop / m * new_m + self.min_intensity

        return drop
