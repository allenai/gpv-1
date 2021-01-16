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
Download `vision_server` private ssh key and change permission (`rw- --- ---`) to allow git clone
```bash
aws s3 cp s3://ai2-prior-gpv/vision_servers ~/.ssh/
chmod 600 ~/.ssh/vision_servers
eval `ssh-agent`
ssh-add ~/.ssh/vision_servers
git clone git@github.com:allenai/gpv.git
```

# Download data
Download COCO images
```bash
bash setup_data.sh
```


