import os
import hydra
import wget
from zipfile import ZipFile

import utils.io as io

@hydra.main(config_path='../../configs',config_name='data/download.yaml')
def main(cfg):
    print(cfg.pretty())
    if cfg.download_coco_images_only:
        data_type_dir = os.path.join(cfg.exp_dir,'coco/images')
        io.mkdir_if_not_exists(data_type_dir)
        for url in cfg.urls['coco']['images'].values():
            filename = url.split('/')[-1]
            filepath = os.path.join(data_type_dir,filename)
            print(filepath)
            if not os.path.exists(filepath):
                print(f'Downloading coco/images/{url}')
                wget.download(url,out=data_type_dir)
                print(' [Done]')
                print(f'Extracting from {filepath}')
                with ZipFile(filepath,'r') as zipobj:
                    zipobj.extractall(data_type_dir)

    for dataset in ['coco','vqa']: #cfg.urls.keys():
        dataset_dir = os.path.join(cfg.exp_dir,dataset)
        for data_type in cfg.urls[dataset].keys():
            data_type_dir = os.path.join(dataset_dir,data_type)
            io.mkdir_if_not_exists(data_type_dir,recursive=True)
            for url in cfg.urls[dataset][data_type].values():
                filename = url.split('/')[-1]
                filepath = os.path.join(data_type_dir,filename)
                print(filepath)
                if not os.path.exists(filepath):
                    print(f'Downloading {dataset}/{data_type}/{url}')
                    wget.download(url,out=data_type_dir)
                    print(' [Done]')
                    print(f'Extracting from {filepath}')
                    with ZipFile(filepath,'r') as zipobj:
                        zipobj.extractall(data_type_dir)



if __name__=='__main__':
    main()

