import os
import copy
import hydra
import random
from tqdm import tqdm
from PyDictionary import PyDictionary
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
import utils.io as io

ALIASES = {
    'person': ['people','man','woman','men','women'],
    'surfboard': ['surf board'],
    'tv': ['television'],
    'toothbrush': ['tooth brush'],
    'dining table': ['dinner table','table'],
    'motorcycle': ['motor cycle','motor bike'],
    'handbag': ['hand bag','bag'],
    'backpack': ['back pack'],
    'airplane': ['air plane','aeroplane'],
    'couch': ['sofa'],
    'skateboard': ['skate board'],
    'snowboard': ['snow board'],
    'toilet': ['toilette'],
    'sports ball': ['ball','football','basketball'],
    'hair drier': ['hair dryer','dryer','drier'],
    'baseball glove': ['glove'],
}

def wnsynonyms(word):
    syn = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syn.add(lemma.name())
    return syn

@hydra.main(config_path='../../configs',config_name='data/coco_synonyms.yaml')
def main(cfg):
    print(cfg.pretty())
    category_split = io.load_json_object(cfg.coco_category_split)
    categories = set()
    for split in category_split.values():
        for category in split:
            categories.add(category['name'])

    dictionary = PyDictionary()
    synonyms = {}
    for category in categories:
        category_synonyms = wnsynonyms(category)#set(dictionary.synonym(category))
        category_synonyms = {' '.join(syn.split('_')).lower() for syn in category_synonyms}
        category_synonyms.add(category)
        if category in ALIASES:
            category_synonyms.update(ALIASES[category])
            
        for syn in copy.deepcopy(category_synonyms):
            tokens = syn.split(' ')
            if len(tokens) > 1:
                category_synonyms.add(''.join(tokens))
                category_synonyms.add('-'.join(tokens))

        synonyms[category] = list(category_synonyms)

    io.dump_json_object(
        synonyms,
        os.path.join(cfg.exp_dir,'coco_category_synonyms.json'))

if __name__=='__main__':
    main()