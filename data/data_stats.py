import hydra

import utils.io as io

@hydra.main(config_path=f'../configs',config_name=f"data/data_stats")
def main(cfg):
    task_cfg = cfg.task_configs
    num_samples = {}
    for dataset in cfg.datasets:
        num_samples[dataset] = []
        for subset in ['train','val','test']:
            if subset=='test' and task_cfg.data_split=='original_split' and dataset not in ['coco_vqa','coco_captioning','refcocop']:
                continue

            samples = io.load_json_object(task_cfg[dataset]['samples'][subset])

            N = len(samples)
            num_samples[dataset].append(N)

            if subset=='test':
                unseen = 0
                for sample in samples:
                    if len(sample['coco_categories']['unseen']) > 0:
                        unseen += 1

                num_samples[dataset].extend([N-unseen, unseen])
        print(num_samples)
    

if __name__=='__main__':
    main()
