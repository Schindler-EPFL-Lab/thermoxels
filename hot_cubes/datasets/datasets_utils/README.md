# How to create a (reflective) dataset

We complete instructions given on [Rebel Nerf](https://github.com/SchindlerEPFL/rebel-nerf/tree/main/scripts/process_thermal)

## Record data

Use the Flir One camera and its app to record thermal images all over the object of interest. 
Try to avoid unwanted reflections and fingers on the images.
In the case of reflective datasets, you can either fix the temperature range prior to the recording session or re-scale the images later. 
To record the data, try to focus on reflections you can control. 
Then record the training datasets with reflections and remove the reflection source in the testing images. 
Clearly separate training picture from testing ones.   

You can use FLIR cloud service to transfer the images to your computer:
- First, select all the images and download them, this will save the msx images in a zip file.
- Then, download one by one the digital images by opening them in the gallery.

This is necessary to obtain perfect alignement between rgb and thermal images as the FLIR extractor misalign them.

Prepare a train folder with the following subfolder:
- msx_train in which you put the msx training images you downloaded from the cloud
- thermal_train : empty for now
- csv_train : empty for now
- rgb_train : empty for now
- msx_eval in which you put the msx testing images you downloaded from the cloud
- thermal_eval : empty for now
- csv_eval : empty for now
- rgb_eval : empty for now


## Processing thermal images :

To process thermal images of Flir One camera, we mainly use the [Flir Extractor Repo](https://github.com/ITVRoC/FlirImageExtractor). 
It extracts raw information of the images and outputs raw thermal images, RGB images, and CSV files with the exact temperature value at every pixel. 

We build on top of that to extract our own temperature images that directly encode the temperature values in greyscale images. 
To do that, we normalise the images based on the maximum and minimum temperatures in the entire dataset.

We also provide a visualiser for the greyscale images that shows the temperature value of every pixel once it's hovered on with the mouse.


## Instructions

To get your normalised thermal images, first create a folder and insert all your images from the Flir One app in it. 

Then, clone the repo in your root directory as follows: 


```bash
 git clone https://github.com/ITVRoC/FlirImageExtractor.git 
```

Make sure to install the following packages:

```bash
sudo dnf install perl-Image-ExifTool.noarch
sudo pip install numpy matplotlib pillow 
```


To create the CSV files, rgb images, and the greyscale raw temperature images, run 
the following using the subfolder adresses you created earlier.

```bash
python scripts/process_thermal/preprocess_thermal.py --path-to-thermal-images <path_to_thermal> --path-to-thermal-images-curated <path_to_output_thermal> --path-to-rgb <path_to_rgb> --path-to-csv-files  <path_to_csv>
```

Do it for both the train and eval folders.

Then replace RGB images from both train and eval 'rgb' folder with the ones you downloaded one by one. 
RGB and thermal images are now perfectly aligned.

It should be noticed that in the FLIR app, there is an option to manually lock the temperature range when taking the photos.
Therefore, if you have locked the temperature range when taking the photos, there is no need to rescale them. 
Do check the temperature range alignment everytime you create a dataset!

In the case eval images are not in the same temperature range as train images, you can re-scale the testing images to the 
training scale using `rescale_thermal_images.py` script with the temperature bounds from 
training and testing sets:

```bash
python hot_cubes/datasets/datasets_utils/rescale_thermal_images.py --input-folder <input_folder> --output-folder <output_folder> --t-min <t_min> --t-max <t_max> --t-min-new <t_min_new> --t-max-new <t_max_new>
```

To be noticed, with Flir, there is an option to manually lock the temperature range when taking the photos.
Therefore, if you have locked the temperature range when taking the photos, there is no need to rescale them.

Then, you can rename the files using the `rename_files.py` script. 


```bash
python hot_cubes/datasets/datasets_utils/rename_files.py --path-to-folder <path_to_folder>
```

Clean the unwanted folder and keep images and thermal ones.

To run COLMAP on these, you need to upload the training and testing images to azure 
blob storage using rebel nerf repo:
```bash
python rebel-nerf/scripts/azurev2/upload_dataset.py --dataset-name <dataset_name> --version <version> --dataset-path <dataset_path> --description <description>
```
Make sure the eval images are in a seperate dataset called `train_dataset_name`+"-eval".


You can use these train and eval folders to run COLMAP : 
```bash
python rebel-nerf/scripts/azurev2/images_to_nerf_dataset_azure.py --scene_name <scene_name> --version <version> --eval_data_version <eval_data_version>
```

Download the output using, the downloaded file is in a `data` folder under the `rebel-nerf` folder:
```bash
python rebel-nerf/scripts/azurev2/download_dataset.py --dataset-name <dataset_name> --version <version>
```
The dataset name can be found in the `Data asset` property of the Outputs in the according created job on Azure.

Finally, use colmap_json_to_txt.py to generate a `pose` folder from `transform.json` file:

```bash
python hot_cubes/datasets/datasets_utils/colmap_json_to_txt.py --folder <folder>
```

Upload your dataset (the downloaded folder) to Azure, it is ready to be used with Plenoxels.
