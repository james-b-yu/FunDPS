import json
import os
from typing import Literal
import numpy as np
from datasets import load_from_disk, DatasetDict
from torch.utils.data import Dataset

from training.dataset_utils import DatasetNormalizer


def transform_jby21_data(name: Literal["darcy", "ns"], item):
    if name == "darcy":
        res = {
            "data": np.stack((0.5 * item["thresh_a_data"], 0.5 * item["thresh_p_data"]))
        }
    elif name == "ns":
        res = {
            "data": np.stack((1.907 * item["a"], 1.95 * item["u"]))
        }
    else:
        raise ValueError()

    if res["data"].ndim == 4:
        res["data"] = res["data"].transpose([1, 0, 2, 3])

    return res


class PDEDataset(Dataset):
    def __init__(
        self,
        path,  # Path to the HuggingFace dataset directory
        phase: Literal["train", "valid", "test"],
        offset=None,  # Start from a specific index.
        # Ensure specific resolution, None = highest available.
        resolution=None,
        max_size=None,  # Maximum number of images to load, None = all images.
        shuffle=False,  # Shuffle the dataset if max_size is set.
        # Enable conditioning labels? False = label dimension is zero.
        use_labels=False,
        xflip=False,  # Augment with horizontal flips.
        cache=False,  # Cache images in CPU memory? int = cache up to N bytes.
    ):
        assert use_labels is False, "Labels are not supported"
        assert xflip is False, "Horizontal flips are not supported"
        if cache:
            print(
                "Warning: CPU memory caching conflicts with multiprocessing, disabling cache")
        if max_size is None and shuffle:
            print("Warning: shuffle=True has no effect when max_size=None")

        self._path = path
        self._name = os.path.basename(path)
        self._offset = offset if offset is not None else 0
        self._normalizer = None

        # Load the HuggingFace dataset
        self._dataset = load_from_disk(self._path)
        if isinstance(self._dataset, DatasetDict):
            self._dataset = self._dataset[phase]
        self._dataset.set_format("numpy", dtype=np.float64)

        # Limit the dataset size
        if max_size is not None:
            assert max_size <= len(
                self._dataset), "max_size must be less than or equal to the number of items"
            if shuffle:
                indices = np.random.choice(
                    len(self._dataset), max_size, replace=False)
                self._dataset = self._dataset.select(indices)
            elif max_size < len(self._dataset):
                self._dataset = self._dataset.select(
                    range(self._offset, self._offset + max_size))
                self._offset = 0

        # Get raw shape from the first item
        first_item = transform_jby21_data(
            "darcy" if path[5] == "d" else "ns", self._dataset[0])
        self._raw_shape = [len(self._dataset)] + list(first_item["data"].shape)

        # Handle resolution downsampling if needed
        self._downsample = 1
        if resolution is not None:
            assert self._raw_shape[2] == self._raw_shape[3]
            assert self._raw_shape[2] % resolution == 0, "Resolution must be divisible by the image resolution"
            if self._raw_shape[2] != resolution:
                self._downsample = self._raw_shape[2] // resolution
                self._raw_shape[2] = self._raw_shape[3] = resolution
                print(f"Downsampling images by a factor of {self._downsample}")

        # Load metadata
        self._metadata = json.load(
            open(os.path.join(self._path, "metadata.json"), "r"))
        if "__version__" not in self._metadata:
            self._metadata["__version__"] = "1.0"
        print("Dataset version:", self._metadata["__version__"])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[self._offset + int(idx)]
        image = transform_jby21_data(
            "darcy" if self._path[5] == "d" else "ns", item)["data"]

        # Handle downsampling if needed
        if self._downsample > 1:
            image = image[..., :: self._downsample, :: self._downsample]

        # Return image and dummy label (for compatibility)
        return image, np.zeros(0)

    def create_normalizer(self):
        normalizer = DatasetNormalizer(
            self._metadata["name"], self._metadata["stats"])
        return normalizer

    def denormalize(self, x_normalized):
        if self._normalizer is None:
            # initializing normalizer earlier causes serialization issues
            self._normalizer = self.create_normalizer()
        return self._normalizer.denormalize(x_normalized)

    @property
    def name(self):
        """Name of the dataset."""
        return self._name

    @property
    def image_shape(self):
        """Shape of each image (CHW format)."""
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        """Number of channels in each image."""
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        """Resolution of the images."""
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        """Shape of labels."""
        return [0]

    @property
    def label_dim(self):
        """Dimension of labels."""
        return 0

    @property
    def has_labels(self):
        """Whether the dataset has labels."""
        return False

    @property
    def has_onehot_labels(self):
        """Whether the dataset has one-hot labels."""
        return False
