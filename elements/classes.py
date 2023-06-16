import itertools
from copy import deepcopy
from typing import List
import warnings

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from elements.shapes import square, circle, triangle, cross, plus
from elements.colors import color_adjustment, colors
from elements.textures import (
    random_spots,
    regular_spots,
    polka,
    chequerboard,
    striped_diagonal,
    striped_vertical,
    striped_horizontal,
    striped_diagonal_alt,
)

SEED_MAX = 1000000

shapes = {
    "square": square,
    "circle": circle,
    "triangle": triangle,
    "cross": cross,
    "plus": plus,
}
textures = {
    "spots": {
        "random": random_spots,
        "regular": regular_spots,
        "polka": polka,
        "chequerboard": chequerboard,
    },
    "stripes": {
        "horizontal": striped_horizontal,
        "vertical": striped_vertical,
        "diagonal": striped_diagonal,
        "diagonal_alt": striped_diagonal_alt,
    },
}


class Element(Dataset):
    def __init__(self, size, shape, color, texture=None, color_seed=None, texture_seed=None):
        self.size = size
        self.shape = shape
        self.color = color
        self.texture = texture
        self.color_seed = color_seed
        self.texture_seed = texture_seed

        self.config = {k: v for k, v in vars(self).items()}

        self.img = shapes[shape](size)

        # Apply texture
        if texture == "solid":
            texture = None
        if texture is not None:
            texture_parts = texture.split("_")
            texture_fn = textures[texture_parts[0]]["_".join(texture_parts[1:])]
            self.img = texture_fn(self.img, seed=texture_seed)

        # Add colour
        self.color_value = color_adjustment(color_seed)
        self.img = (self.img.reshape((size, size, 1)) * (self.color_value * colors[color])).astype(np.uint8)

        # Set all black pixels to white
        self.img[(self.img == [0, 0, 0]).all(axis=2)] = [255, 255, 255]

    def __len__(self):
        return len(self.img)

    def __repr__(self):
        return "Element <" + ", ".join([f"{k}: {v}" for k, v in self.config.items()]) + ">"

    def belongs_to_class(self, class_config):
        belongs = True
        for k, v in class_config.items():
            if v is None:
                continue
            else:
                if self.config[k] != v:
                    belongs = False
                    break
        return belongs


class ElementImage:
    def __init__(self, elements: List[Element], size: int = 224, loc_seed: int = None):
        self.elements = elements
        self.size = size
        self.loc_seed = loc_seed
        self.class_labels = None
        self.class_labels_oh = None

        self.img = np.zeros((size, size, 3), dtype=np.uint8) + 255
        self.locs = self.choose_locations(size, elements, loc_seed)

        self.config = {
            "elements": [v.config for v in self.elements],
            "size": self.size,
            "loc_seed": self.loc_seed,
        }
        self.info = {
            **self.config,
            "locs": self.locs,
            "class_labels": self.class_labels,
            "class_labels_oh": self.class_labels_oh,
        }

        for i in range(len(elements)):
            self.img = self.place_element(self.img, self.elements[i], self.locs[i])

    @staticmethod
    def place_element(canvas, element, loc):
        canvas[loc[0] : loc[0] + len(element), loc[1] : loc[1] + len(element)] = element.img
        return canvas

    def __len__(self):
        return len(self.img)

    def __repr__(self):
        return "ElementImage <" + ", ".join([f"{k}: {v}" for k, v in self.config.items()]) + ">"

    def belongs_to_class(self, class_config):
        for element in self.elements:
            if element.belongs_to_class(class_config):
                return True
        return False

    def update_class_labels(self, class_configs):
        classes = []
        for i, config in enumerate(class_configs):
            if self.belongs_to_class(config):
                classes.append(i)
        self.class_labels = classes
        self.class_labels_oh = np.eye(len(class_configs))[classes].sum(axis=0)
        self.info["class_labels"] = self.class_labels
        self.info["class_labels_oh"] = self.class_labels_oh
        return classes

    @staticmethod
    def from_config(config):
        # Only select relevant keys
        # Allows self.info to be fed in as well as self.config
        config = {k: config[k] for k in ["elements", "size", "loc_seed"]}
        elements = [Element(**v) for v in config["elements"]]
        config["elements"] = elements
        return ElementImage(**config)

    @staticmethod
    def choose_locations(canvas_size, elements, seed=None, max_attempts=100, raise_error=False):
        rng = np.random.default_rng(seed)
        element_sizes = [len(element) for element in elements]
        locations = []
        for element_size in element_sizes:
            placed = False
            attempts = 0
            while (not placed) and (attempts < max_attempts):
                x, y = rng.integers(0, canvas_size - element_size, 2)
                attempts += 1

                overlap = False
                for i, coord in enumerate(locations):
                    if (
                        (x < coord[0] + element_sizes[i])
                        and (x + element_size > coord[0])
                        and (y < coord[1] + element_sizes[i])
                        and (y + element_size > coord[1])
                    ):
                        overlap = True
                        break
                if overlap:
                    # If it overlaps, try a different position
                    continue

                locations.append((x, y))
                placed = True
            if not placed:
                if raise_error:
                    raise ValueError("Not all elements could be placed without overlapping!")
                else:
                    remaining_elements = len(element_sizes) - len(locations)
                    warnings.warn(
                        f"Not all elements could be placed without overlapping! Placing remaining {remaining_elements} elements randomly."
                    )
                    for i in range(remaining_elements):
                        x, y = rng.integers(0, canvas_size - element_sizes[len(locations)], 2)
                        locations.append((x, y))
        return locations


class ElementDataset:
    def __init__(
        self,
        allowed,
        class_configs,
        n,
        img_size,
        element_n,
        element_size,
        element_size_delta,
        element_seed,
        loc_seed,
        allowed_combinations=None
    ):
        self.allowed = allowed
        self.class_configs = class_configs
        self.n = n
        self.img_size = img_size
        self.element_n = element_n
        self.element_size = element_size
        self.element_size_delta = element_size_delta
        self.element_seed = element_seed
        self.loc_seed = loc_seed
        self.allowed_combinations = allowed_combinations

        self.config = {k: v for k, v in vars(self).items()}
        self.allowed["sizes"] = (self.element_size - self.element_size_delta,
                                 self.element_size + self.element_size_delta)

        self.element_rng = np.random.default_rng(self.element_seed)
        self.loc_rng = np.random.default_rng(self.loc_seed)

        self.element_seeds = self.element_rng.integers(0, SEED_MAX, self.n)
        self.loc_seeds = self.loc_rng.integers(0, SEED_MAX, self.n)

        self.transform = T.Compose([
            T.ToTensor(),
            ])

    def __getitem__(self, idx):
        item = self.get_item(idx)
        img = self.transform(item.img)
        return [img, item.class_labels_oh]

    def get_item(self, idx):
        element_configs = self.choose_element_configs(
            self.element_n,
            self.allowed["sizes"],
            self.allowed["shapes"],
            self.allowed["colors"],
            self.allowed["textures"],
            self.element_seeds[idx],
            allowed_combinations=self.allowed_combinations
        )
        elements = [Element(**v) for v in element_configs]
        img = ElementImage(elements, self.img_size, self.loc_seeds[idx])
        img.update_class_labels(self.class_configs)
        return img

    def __len__(self):
        return self.n

    @staticmethod
    def choose_element_configs(n, allowed_sizes, allowed_shapes, allowed_colors, allowed_textures, seed=None, allowed_combinations=None, max_iter=1000):
        rng = np.random.default_rng(seed)

        if allowed_combinations is None:
            allowed_combinations = list(itertools.product(allowed_shapes, allowed_colors, allowed_textures))

        configs = []
        i = 0
        while len(configs) < n:
            i += 1
            config = dict(
                size=rng.integers(*allowed_sizes),
                shape=rng.choice(allowed_shapes),
                color=rng.choice(allowed_colors),
                color_seed=rng.integers(SEED_MAX),
                texture=rng.choice(allowed_textures),
                texture_seed=rng.integers(SEED_MAX),
            )
            if (config["shape"], config["color"], config["texture"]) in allowed_combinations:
                configs.append(config)
            if i > max_iter:
                raise TimeoutError(f"Only {len(configs)} out of {n} chosen "
                                   f"in the max no. iterations ({max_iter})")
        return configs

    def save_imgs(self, prefix, indices=None):
        if indices is None:
            indices = range(self.n)

        for i in indices:
            img = self.get_item(i)
            img = Image.fromarray(img.img)
            img.save(f"{prefix}{i:04d}.jpg")
