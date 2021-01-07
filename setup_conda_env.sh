#conda create -n gpv python=3.6 -y
#conda activate gpv

################################################################################
# conda installs
################################################################################
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge scikit-image -y
conda install -c conda-forge spacy -y
conda install -c conda-forge spacy-lookups-data -y
conda install h5py -y
python -m spacy download en_core_web_sm


################################################################################
# pip installs
################################################################################
pip install hydra-core --upgrade --pre
pip install ipython
pip install ipdb
pip install tensorboard
pip install opencv-python==4.2.0.32
pip install tqdm
pip install transformers
pip install nltk
pip install wget
pip install PyDictionary
pip install git+https://github.com/salaniz/pycocoevalcap
pip install lmdb
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install boto3
pip install tensorboardX
pip install pytorch_transformers==1.0.0
pip install imagesize