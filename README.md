# Oil Riggery

## Objective

Count the number of oil rigs in given areas of the world from satellite imagery.

## Data

Annotated data:
- [An Oil Well Dataset Derived from Satellite-Based Remote Sensing](https://www.mdpi.com/2072-4292/13/6/1132#) called NEPU_OWOD-1.0.

Real-world data for evaluation:
- [Aerial Image Dataset](https://captain-whu.github.io/AID/) a large scale dataset for satellite image classification.
- Create my own dataset. Multi-step process including 1. Using QGIS to create a grid of points over the world. 2. Using Google Earth Engine to download satellite imagery for each point. 3. Using LabelImg to annotate the downloaded images.

## Data exploration

A streamlit data explorer is provided under `/data`. Pre-requisites:

* Add the `NEPU_OWOD-1.0` under the `/data` folder. It can be downloaded following the link above.

Run with:

```bash
poetry run streamlit run oil_riggery/data/Data_explorer.py
```

## Method

TBD

