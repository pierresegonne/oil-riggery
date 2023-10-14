from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

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

    train_test_eval_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    train_indices: np.ndarray
    test_indices: np.ndarray
    eval_indices: np.ndarray

    @staticmethod
    def load() -> RCNNDataset:
        """
        Load the dataset from disk and stores it in memory.
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



if __name__ == "__main__":
    dataset = NEPUDataset.load(
        "data/NEPU_OWOD-1.0/JPEGImages",
        "data/NEPU_OWOD-1.0/Annotations",
    )