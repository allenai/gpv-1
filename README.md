# General Purpose Vision

# Install dependencies

## Setup aws cli
Download and install awscli
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Configure aws cli
```
aws configure
```
Set region to `us-west-2` and output to `json`.

## Setup conda env
Create conda environment
```bash
conda create -n gpv python=3.6 -y
conda activate gpv
```

Install libraries
```bash
bash setup_conda_env.sh
```

## Setup git
Note - these are temporary instructions to be deleted later
Download `vision_server` private ssh key and change permission (`rw- --- ---`) to allow git clone
```bash
aws s3 cp s3://ai2-prior-gpv/vision_servers ~/.ssh/
chmod 600 ~/.ssh/vision_servers
eval `ssh-agent`
ssh-add ~/.ssh/vision_servers
git clone --recurse-submodules git@github.com:allenai/gpv.git
```

# Paths

Decide the following paths:
- `<data_dir>`: This is the directory where images and annotations will be saved
- `<output_dir>`: This is where outputs of various experiments will be saved including model checkpoints, visualization, inference and evaluation results

`<data_dir>` and `<output_dir>` refer to these absolute paths in the instructions below. 

# Download data
Download splits, detr ckpts, and COCO images. `data_dir` is the path to the directory where you want the data downloaded.  
```bash
bash setup_data.sh <data_dir>
```

# Download model and evaluate

| Model | Split | Link |
|-------|-------|------|
| GPV | COCO | [Download](https://ai2-prior-gpv.s3-us-west-2.amazonaws.com/public/trained_models/gpv_all_original_split/ckpts/model.pth) |
| GPV | COCO-SCE | [Download](https://ai2-prior-gpv.s3-us-west-2.amazonaws.com/public/trained_models/gpv_all_gpv_split/ckpts/model.pth) |

To use any of these models, download them into `<output_dir>/<exp_name>/ckpts` directory as follows:
```
wget <link> -P <output_dir>/<exp_name>/ckpts/
```
`<exp_name>` could be any directory name of your choice such as `gpv_coco` or `gpv_coco_sce`.

To evaluate on any one of `CocoClassification`, `CocoVqa`, `CocoDetection` (refered to as the Localization task in the paper), `CocoCaptioning`, or `RefCocop` run the following after updating `<output_dir>` and `<data_dir>` in `configs/exp/gpv.yaml`
```
bash exp/gpv/eval.sh <exp_name> <task_name> <subset> <split>
```

- `<task_name>`: set to `all` to evaluate on all 5 tasks, `all_but_refexp` to evalute on all tasks excepts RefCocop, or the name of tasks to evaluate only on that task.
- `<split>`: set to `original_split` (COCO) or `gpv_split` (COCO-SCE). 
- `<subset>`: set to `train` or `val` for COCO (no `test` since COCO test annotations are hidden) and `train`, `val`, or `test` for COCO-SCE.

# Train model

We provide scripts for training GPV on one or more of the following task: 
- `CocoClassification`
- `CocoVqa`
- `CocoDetection` (refered to as the Localization task in the paper)
- `CocoCaptioning`

Training GPV-1 involves 3 steps:
- **Step 1:** Update the `configs/exp/gpv.yaml` file. Here are the key parameters to consider (the ones marked with a star can be set later in Step 3):
    - `num_gpus_per_node` (set to 4 if you have 24GB GPUs, 2 for 48GB and 1 for 80GB)
    - `dist_url`
    - `output_dir` *
    - `data_dir` *
    - `model.pretr_detr` *
- **Step 2:** Decide the dataset or combination of supported datasets to train the model. This is specified through one of the files in `configs/learning_datasets`. For instance, `all.yaml` trains on all 4 tasks, `cap_vqa.yaml` trains on `CocoCaptioning` & `CocoVqa`, and `cap.yaml` trains only on `CocoCaptioning`. If you don't see a dataset combination you may add one by modifying `all.yaml`. We refer to this as `<learning_datasets>`
- **Step 3:** Launch training using `exp/gpv/scripts/train_gpv.sh` as follows:
    ```
    bash exp/gpv/scripts/train_gpv.sh <learning_datasets> <data_split> <exp_name> <output_dir> <data_dir>
    ```
    Note that training comprises of 2 sub-steps. First, the model is trained for `training.frozen_epochs` (in `configs/exp/gpv.yaml`) steps with DETR weights frozen. Then the model is finetuned end-to-end for a total of `training.num_epochs` epochs. `train_gpv.sh` executes both steps sequentially. 