import cv2
import numpy as np
import torch

from . import DropletTransform


class DropletFG(DropletTransform):

    def __init__(self, tresh : float = 0.15, linspace: dict = {"theta":100,"r":1000}, tol_imbalance : float = 0.1) -> None:
        super().__init__()
        self.tresh = tresh
        self.linspace = linspace
        self.tol_imbalance = tol_imbalance

    def transform(self, drop: torch.FloatTensor) -> torch.FloatTensor:
        
        mid_x, mid_y = drop.shape[0] // 2 - 1, drop.shape[1] // 2 - 1

        # For different angles, compute radiuses at where a ray from the center of the image 
        # goes bellow the given threshold; this signals the end of the droplet
        idx = []
        rads = []
        mask = drop < self.tresh
        for theta in np.linspace(0, 2 * np.pi, self.linspace["theta"]):
            for r in np.linspace(0.3, 1.0, self.linspace["r"]):
                x = round(mid_x *  (1 + r * np.cos(theta)))
                y = round(mid_y *  (1 + r * np.sin(theta)))
                
                if mask[x, y]:
                    idx.append((y, x))
                    rads.append(r)
                    break

        # Select valid points as the found radiuses withing some quantiles
        lower = np.quantile(rads, 0.05)
        upper = np.quantile(rads, 0.95)
        valid_points = np.array([i for i, r in zip(idx, rads) if lower <= r and r <= upper])

        # Compute convex hull of valid points
        hull = cv2.convexHull(valid_points)
        convex_mask = np.zeros_like(drop)
        cv2.fillPoly(convex_mask, pts = [hull[:, 0]], color=1)
        convex_mask = convex_mask.astype(bool)
        drop[~convex_mask] = 0

        # Only keep a bounding box around the droplet

        # Compute x-box
        x_valid = [i for i in range(convex_mask.shape[0]) if np.any(convex_mask[i, :])]
        x_min, x_max = np.min(x_valid), np.max(x_valid)

        # Compute y-box
        y_valid = [i for i in range(convex_mask.shape[1]) if np.any(convex_mask[:, i])]
        y_min, y_max = np.min(y_valid), np.max(y_valid)

        # Shrink droplet
        drop = drop[
            x_min:x_max,
            y_min:y_max,
        ]

        # Ensure a certain balance
        if self.tol_imbalance is not None:
            ratio = abs(drop.shape[0] - drop.shape[1]) / max(drop.shape[0], drop.shape[1])
            assert ratio < self.tol_imbalance

        return drop
