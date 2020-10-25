import os
from collections import Counter
import hydra
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import utils.io as io
from datasets.coco_multitask_dataset import CocoMultitaskDataset


@hydra.main(config_path=f'../configs',config_name=f"data/coco_vocab")
def main(cfg):
    vocab = set()
    cnt = Counter()
    vocab.update(['__pad__','__cls__','__stop__','__unk__'])
    for subset in ['train','val','test']:
        tasks = CocoMultitaskDataset(
            cfg.learning_datasets,cfg.task_configs,subset)
        print('Subset:',subset)
        for i in tqdm(range(len(tasks))):
            query,targets = tasks[i]
            if 'answer' in targets:
                tokens = word_tokenize(targets['answer'].lower())
                vocab.update(tokens)
                for token in tokens:
                    cnt[token] += 1
            
            if i%10000==0:
                msg = 'Vocab size: {}'.format(len(vocab))
                tqdm.write(msg)
                selected_size = 0
                for word in cnt:
                    if cnt[word] > cfg.min_count:
                        selected_size += 1
                msg = 'Selected vocab size: {}'.format(selected_size)
                tqdm.write(msg)

        
    selected_vocab = []
    for word in vocab:
        if cnt[word] > cfg.min_count:
            selected_vocab.append(word)
    
    print('Selected Vocab:',len(selected_vocab))
    io.dump_json_object(
        sorted(selected_vocab),
        os.path.join(cfg.exp_dir,'vocab.json'))
    
    io.dump_json_object(
        cnt,
        os.path.join(cfg.exp_dir,'vocab_count.json'))


if __name__=='__main__':
    main()