from pathlib import Path
from glob import glob
import streamlit as st
import untangle
import cv2


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


DATASET_BASE_PATH = Path(__file__).parent.parent.resolve() / "NEPU_OWOD-1.0"
BASE_IMAGE_PATH = DATASET_BASE_PATH / "JPEGImages/"
image_paths = sorted(glob(BASE_IMAGE_PATH.as_posix() + "/*.jpg"))
image_ids = {p.split("/")[-1].split(".")[0]: {"dataset": []} for p in image_paths}

image_set_paths = sorted(
    glob((DATASET_BASE_PATH / "ImageSets/Main").as_posix() + "/*.txt")
)
for set_path in image_set_paths:
    set_name = set_path.split("/")[-1].split(".")[0]
    with open(set_path, "r") as f:
        for image_id in f.readlines():
            image_id = image_id.strip()
            image_ids[image_id]["dataset"].append(set_name)
            image_ids[image_id]["annotations"] = parse_xml_annotation(
                (DATASET_BASE_PATH / "Annotations" / f"{image_id}.xml").as_posix()
            )

st.set_page_config(
    page_title="An Oil Well Dataset Derived from Satellite-Based Remote Sensing",
    page_icon="🛰",
)

st.markdown(
    """
            # NEPU-OWOD-1.0

            _Source_: [An Oil Well Dataset Derived from Satellite-Based Remote Sensing](https://www.mdpi.com/2072-4292/13/7/1319/htm)

            The dataset includes 1192 oil wells in 432 images from Daqing City, which has the largest oilfield in China.


            ## Visualise an image
            """
)
st.sidebar.header("Select an image ID")

image_id = st.sidebar.selectbox("Image ID", image_ids.keys(), index=0)

# Show image with bounding boxes
img = cv2.imread((BASE_IMAGE_PATH / f"{image_id}.jpg").as_posix())
for obj in image_ids[image_id]["annotations"]["objects"]:
    cv2.rectangle(
        img,
        (obj["xmin"], obj["ymin"]),
        (obj["xmax"], obj["ymax"]),
        (0, 255, 0),
        2,
    )
st.image(img, use_column_width=True)

# Describe image
st.write(f"Image ID: {image_id}")
st.write(f"Dataset: {image_ids[image_id]['dataset']}")
st.write(f"Annotations: {image_ids[image_id]['annotations']}")
