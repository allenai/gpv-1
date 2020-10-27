hostname=172.16.0.172
sshkey=$1

## copy splits
mkdir /home/tanmayg/Data/gpv/learning_phase_data
scp -r -i $sshkey tanmayg@$hostname:/home/tanmayg/Data/gpv/learning_phase_data/splits.zip /home/tanmayg/Data/gpv/learning_phase_data
unzip /home/tanmayg/Data/gpv/learning_phase_data/splits.zip -d /home/tanmayg/Data/gpv/learning_phase_data/

## copy detr
detr_pth=/home/tanmayg/Data/gpv/coco_exp/detr-r50-e632da11.pth
mkdir /home/tanmayg/Data/gpv/coco_exp
scp -i $sshkey tanmayg@$hostname:$detr_pth /home/tanmayg/Data/gpv/coco_exp