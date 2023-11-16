# DAML

Create a Conda environment:

```bash
conda env create -f environment.yml
```
This will create a new Conda environment based on the environment.yml file.

Activate the Conda environment:
```
conda activate your_environment_name
```
# Dataset
Stanford Cars Dataset (Cars196)

-- Download from (https://ai.stanford.edu/~jkrause/cars/car_dataset.html) or use datasets/cars196_downloader.py. 

-- Convert to hdf5 file using cars196_converter.py.

-- Put it in datasets/data/cars196/cars196.hdf5.

# Pretrained model
GoogleNet V1 pretrained model can be downloaded from (https://github.com/Wei2624/Feature_Embed_GoogLeNet)

# Usage
## For Cars196 dataset:

```bash
python main_npair.py --dataSet='cars196' --batch_size=128 --Regular_factor=5e-3 --init_learning_rate=7e-5 --load_formalVal=False --embedding_size=128 --loss_l2_reg=3e-3 --init_batch_per_epoch=500 --batch_per_epoch=64 --max_steps=8000 --beta=1e+4 --lr_gen=1e-2 --num_class=99 --_lambda=0.5 --s_lr=1e-3
```



