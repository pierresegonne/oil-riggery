import shutil

from oil_riggery.src.dataset import NEPUDataset

""" poetry run python oil_riggery/src/datasets/NEPU/loader.py """

dataset = NEPUDataset.load(
    "oil_riggery/data/NEPU_OWOD-1.0/JPEGImages",
    "oil_riggery/data/NEPU_OWOD-1.0/Annotations",
)

# Move images to oil_riggery/src/datasets/NEPU/images for their respective train/test/val folders
copy_images = True
if copy_images:
    for fn in [dataset.filenames[i] for i in dataset.train_indices]:
        shutil.copy(
            f"oil_riggery/data/NEPU_OWOD-1.0/JPEGImages/{fn}",
            f"oil_riggery/src/datasets/NEPU/images/train/{fn}",
        )
    for fn in [dataset.filenames[i] for i in dataset.test_indices]:
        shutil.copy(
            f"oil_riggery/data/NEPU_OWOD-1.0/JPEGImages/{fn}",
            f"oil_riggery/src/datasets/NEPU/images/test/{fn}",
        )
    for fn in [dataset.filenames[i] for i in dataset.eval_indices]:
        shutil.copy(
            f"oil_riggery/data/NEPU_OWOD-1.0/JPEGImages/{fn}",
            f"oil_riggery/src/datasets/NEPU/images/val/{fn}",
        )

"""
YOLOV5 dataset format is that for each image we should have a .txt file
Where each line is a bounding box with the format:
$label $x_center $y_center $width $height
where fields are normalised by the image width and height
"""
for i, fn in enumerate(dataset.filenames):
    is_train, is_eval, is_test = (
        i in dataset.train_indices,
        i in dataset.eval_indices,
        i in dataset.test_indices,
    )
    image_dataset_type = "train" if is_train else "test" if is_test else "val"
    annotations = [a for a in dataset.annotations if a.filename == dataset.filenames[i]]
    label = 0
    img = dataset._load_image(fn, pad_and_resize=False)
    img_width, img_height = img.shape[1], img.shape[0]
    for annotation in annotations:
        bb_box = dataset._load_target(annotation, pad_and_resize=False)
        bb_x_center = (bb_box[0] + bb_box[2]) / 2 / img_width
        bb_y_center = (bb_box[1] + bb_box[3]) / 2 / img_height
        bb_width = (bb_box[2] - bb_box[0]) / img_width
        bb_height = (bb_box[3] - bb_box[1]) / img_height
        with open(
            f"oil_riggery/src/datasets/NEPU/labels/{image_dataset_type}/{fn.split('.')[0]}.txt",
            "a",
        ) as f:
            f.write(f"{label} {bb_x_center} {bb_y_center} {bb_width} {bb_height}\n")
