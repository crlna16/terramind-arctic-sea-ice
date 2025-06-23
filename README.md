## TerraMind Demo

### Ice

Dataset: https://data.dtu.dk/collections/AI4Arctic_Sea_Ice_Challenge_Dataset/6244065

Based on Sentinel-1 and MWR. Any possible to combine ??

Idea: Predict sea ice type

- Separate models for SIC, SOD, FLOE
- SIC could also be considered a pixel-wise regression task
- Explore thinking in modalities

Comparison
---

1. Terramind with fine tuning
2. Terramind with thinking in modalities
3. Training U-Net from scratch


## Other ideas
### Crop type

Idea: Transferability of crop type classification, cloud agnostic, spatial / temporal domain shift
Dataset: https://github.com/MarcCoru/MTLCC

Another dataset: https://zenodo.org/records/14094196

Eurocrops-ML: https://github.com/dida-do/eurocropsml


### Air quality

Idea: predict air quality from satellite data, use land cover as additional reasoning mode??

### Drought

Idea: Predict deviation from typical soil moisture??

### Scene annotation

Dataset: https://arxiv.org/pdf/2207.09507

With seasonal transferability?

https://zenodo.org/records/6979994

