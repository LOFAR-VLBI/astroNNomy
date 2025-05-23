# Step by step for running a model
## Install repository
```shell
pip install -e  git+https://github.com/LOFAR-VLBI/astroNNomy.git#egg=astroNNomy
```

## Get training data
```shell
mkdir <path-to-fits-dir>
cd <path-to-fits-dir>

ONLINE_DATA_PATH=https://public.spider.surfsara.nl/project/lofarvwf/jdejong/CORTEX/calibrator_selection_robertjan/cnn_data/
wget -r --spider --no-parent $ONLINE_DATA_PATH 2>&1 | grep -o 'https://[^ ]*' | grep -E '^.*\/{1}[^\/]+?\.[^\/]+?$' > urls.txt
xargs -n 1 -P 16 wget -q -r -np --no-clobber < urls.txt 
```

## Preprocess fits files and convert to npz (numpy compressed)
This will put the converted `*.npz` files in the same dir as the input `*.fits` files.
```shell
python astronnomy.pre_processing_for_ml <path-to-fits-dir>
```

## (Optional) Copy files to /dev/shm for fast dataloading
```shell
find <path-to-fits-dir> -type f -name "*.npz" | xargs -n 1 -P 8 -i rsync -R {} /dev/shm
```

## Run neural network training
```shell
python -m astronnomy.training.train_nn /dev/shm/<fullpath>
```
The dataloader expect filetree to be in the following format:
```text
<filepath>
  |- continue
  |- stop
  |- continue_val
  |- stop_val
```
