# How to create a (reflective) dataset

To create this dataset, please first follow the instructions given by the repository [Thermo NeRF](https://github.com/Schindler-EPFL-Lab/thermo-nerf/tree/main):

- [Data Collection using a Flir camera](https://github.com/Schindler-EPFL-Lab/thermo-nerf/blob/main/thermo_scenes/docs/Collect_new_dataset.md#data-collection-using-a-flir-camera)
- [Data Extraction](https://github.com/Schindler-EPFL-Lab/thermo-nerf/blob/main/thermo_scenes/docs/Collect_new_dataset.md#data-extraction)

Then please follow the instructions given by [Rebel NeRF](https://github.com/SchindlerEPFL/rebel-nerf/blob/main):

- [Upload extracted datasets on Azure](https://github.com/SchindlerEPFL/rebel-nerf/blob/main/docs/dataset_preparation.md#upload-extracted-datasets-on-azure)
- [Run COLMAP in Azure](https://github.com/SchindlerEPFL/rebel-nerf/blob/main/docs/dataset_preparation.md#run-colmap-in-azure)

Then to generate the poses, follow thes instrauction bellow.

## Generate poses

To generate a `pose` folder from `transform.json` file, use the script `colmap_json_to_txt.py` from the repository [Thermal Voxel](https://github.com/SchindlerEPFL/thermal-voxel/blob/main/thermoxels/datasets/datasets_utils/colmap_json_to_txt.py), with `folder` being the path to the folder of images with transforms.json:

```bash
python thermoxels/datasets/datasets_utils/colmap_json_to_txt.py --folder <folder>
```

Finally follow the instruction  given by [Rebel NeRF](https://github.com/SchindlerEPFL/rebel-nerf/blob/main):

- [Upload final dataset to Azure](https://github.com/SchindlerEPFL/rebel-nerf/blob/main/docs/dataset_preparation.md#upload-final-dataset-to-azure)
