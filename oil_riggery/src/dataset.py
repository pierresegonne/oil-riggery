from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, List, Tuple

import cv2
import numpy as np
import untangle


def parse_xml_annotation(xml_file: str) -> dict:
    xml_doc = untangle.parse(xml_file)
    annotation_dict = {
        "width": int(xml_doc.annotation.size.width.cdata),
        "height": int(xml_doc.annotation.size.height.cdata),
        "depth": int(xml_doc.annotation.size.depth.cdata),
        "objects": [
            {
                "name": obj.name.cdata,
                "xmin": int(obj.bndbox.xmin.cdata),
                "ymin": int(obj.bndbox.ymin.cdata),
                "xmax": int(obj.bndbox.xmax.cdata),
                "ymax": int(obj.bndbox.ymax.cdata),
            }
            for obj in xml_doc.annotation.object
        ],
    }
    return annotation_dict

@dataclass
class Annotation:
    filename: str
    name: None | str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    width: None | int
    height: None | int
    depth: None | int

    def to_target(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float32)

class RCNNDataset:

    annotations_path: str
    files_path: str

    filenames: List[str]
    annotations: List[Annotation]
    size: int

    train_test_eval_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    train_indices: np.ndarray
    test_indices: np.ndarray
    eval_indices: np.ndarray
    seed: int = 658940

    @staticmethod
    def load() -> RCNNDataset:
        """
        Load the dataset from disk and stores it in memory.
        """
        raise NotImplementedError

    def get_train_dataset(self) -> Generator[np.ndarray, np.ndarray]:
        """
        Returns a generator that yields batches of training data.
        """
        raise NotImplementedError

    def get_test_dataset(self) -> Generator[np.ndarray, np.ndarray]:
        """
        Returns a generator that yields batches of testing data.
        """
        raise NotImplementedError

    def get_eval_dataset(self) -> Generator[np.ndarray, np.ndarray]:
        """
        Returns a generator that yields batches of evaluation data.
        """
        raise NotImplementedError

class NEPUDataset(RCNNDataset):

    def __init__(
            self,
            files_path: str,
            annotations_path: str,
            filenames: List[str],
            annotations: List[Annotation],
            ) -> None:
        self.files_path = files_path
        self.annotations_path = annotations_path
        self.annotations = annotations
        self.filenames = filenames

        n_samples = len(self.filenames)
        self.size = n_samples
        n_train = int(n_samples * self.train_test_eval_split[0])
        n_test = int(n_samples * self.train_test_eval_split[1])

        indices = np.arange(n_samples)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:n_train+n_test]
        self.eval_indices = indices[n_train+n_test:]

        super().__init__()

    @staticmethod
    def load(
        files_path: str,
        annotations_path: str,
        file_extension: str = "jpg",
        annotation_extenstion: str = "xml"
    ) -> NEPUDataset:
        # Verify that folders exist
        if not os.path.exists(files_path):
            raise FileNotFoundError(f"Images path {files_path} does not exist.")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations path {annotations_path} does not exist.")

        filenames = [f for f in os.listdir(files_path) if f.endswith(file_extension)]
        filenames = sorted(filenames)

        annotations = []
        for filename in filenames:
            name = filename.split(".")[0]
            annotation_dict = parse_xml_annotation(
                os.path.join(annotations_path, f"{name}.{annotation_extenstion}")
            )
            for obj in annotation_dict["objects"]:
                annotations.append(
                    Annotation(
                        filename=filename,
                        name=obj["name"],
                        xmin=obj["xmin"],
                        ymin=obj["ymin"],
                        xmax=obj["xmax"],
                        ymax=obj["ymax"],
                        width=annotation_dict["width"],
                        height=annotation_dict["height"],
                        depth=annotation_dict["depth"],
                    )
                )

        # Verify consistency - the set of filenames appearing in annotations should be the same as the set of filenames
        # in the images folder.
        annotation_filenames = set([a.filename for a in annotations])
        image_filenames = set(filenames)
        if annotation_filenames != image_filenames:
            raise ValueError(
                f"Annotations and images are not consistent. "
                f"Annotations: {annotation_filenames}. "
                f"Images: {image_filenames}."
            )


        return NEPUDataset(
            files_path=files_path,
            annotations_path=annotations_path,
            filenames=filenames,
            annotations=annotations,
        )

    def _load_image(self, filename: str) -> np.ndarray:
        return (cv2.imread(os.path.join(self.files_path, filename)) / 255.0).astype(np.float32)

    def get_train_dataset(self) -> Generator[np.ndarray, np.ndarray]:
        for i in self.train_indices:
            yield self._load_image(self.filenames[i]), self.annotations[i].to_target()

    def get_test_dataset(self) -> Generator[np.ndarray, np.ndarray]:
        for i in self.test_indices:
            yield self._load_image(self.filenames[i]), self.annotations[i].to_target()

    def get_eval_dataset(self) -> Generator[np.ndarray, np.ndarray]:
        for i in self.eval_indices:
            yield self._load_image(self.filenames[i]), self.annotations[i].to_target()





if __name__ == "__main__":
    dataset = NEPUDataset.load(
        "oil_riggery/data/NEPU_OWOD-1.0/JPEGImages",
        "oil_riggery/data/NEPU_OWOD-1.0/Annotations",
    )
    train_dataset = dataset.get_train_dataset()
    img, annotations = next(train_dataset)