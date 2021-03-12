url=$1
perc=$2
bash exp/gpv_biatt_box_text/scripts/ft_gpv.sh 4 $url gpv_all_gpv_split $perc
bash exp/gpv_biatt_box_text/scripts/ft_gpv.sh 4 $url gpv_det_gpv_split $perc