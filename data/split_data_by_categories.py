import os
import hydra
import random
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from collections import Counter
from data.coco.synonyms import SYNONYMS
import utils.io as io
import spacy
nltk.download('punkt')
# Run this line!
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

class AssignCocoCategories():
    def __init__(self,categories,synonyms):
        self.lemmatizer = WordNetLemmatizer()
        self.categories = categories
        self.synonyms = self.lemmatize_synonyms(synonyms)
        self.special_categories = ['orange', 'dog', 'cup', 'clock']

    def is_subsequence(self,needle,haystack):
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return True
        
        return False

    def lemmatize_tokens(self,tokens):
        return [self.lemmatizer.lemmatize(w.lower()) for w in tokens]

    def lemmatize_synonyms(self,synonyms):
        synonym_stems = {}
        for k,syns in synonyms.items():
            synonym_stems[k] = []
            for syn in syns:
                tokens = word_tokenize(syn)
                synonym_stems[k].append(self.lemmatize_tokens(tokens))
        
        return synonym_stems

    def remove_special(self,category_name,appearance,text_tokens,text):
        assert category_name in self.special_categories
        appearance_indices = [i for i, word in enumerate(text_tokens) if word==appearance]
        # If the category we check is dog,
        # we don't want to catch "hot dog".
        if category_name == 'dog':
            for i in appearance_indices:
                if i == 0 or (i > 0 and text_tokens[i-1] != 'hot'):
                    return False
                return True
        # If the category we check is cup,
        # we don't want to catch "wearing glasses"
        # or "wine glasses".
        if category_name == 'cup':
            if appearance in ['glass', 'glasses'] and \
                    'wine' in text_tokens:
                return True
            for i in appearance_indices:
                if i == 0 or (i > 0 and text_tokens[i-1] not in ['wear', 'wearing']):
                    return False
                return True
        # If the category we check is orange,
        # we don't want to catch "what color --> orange"
        # or orange as anything but a noun.
        if category_name == 'orange':
            spacy_text = nlp(text.lower())
            for spacy_word in spacy_text:
                if spacy_word.text in ['orange', 'oranges'] \
                        and spacy_word.pos_ == 'NOUN':
                    return False
            # only return True if no "orange" was a noun:
            return True
        # If the category we check is clock,
        # we don't want to catch "watch" as anything but a noun.
        if category_name == 'clock' and appearance in ['watch', 'watches']:
            spacy_text = nlp(text.lower())
            if spacy_text[-1].text in ['watch', 'watches']:
                return False
            for spacy_word in spacy_text:
                if spacy_word.text in ['watch', 'watches'] \
                        and spacy_word.pos_ == 'NOUN':
                    return False
            # only return True if no "watch" was a noun:
            return True
        return

    def assign(self,sample):
        text = sample['query']
        if 'answer' in sample:
            text = text + ' ' + sample['answer']
        text_tokens = self.lemmatize_tokens(word_tokenize(text))
        assigned_categories = []
        for category in self.categories:
            for syn in self.synonyms[category['name']]:
                if self.is_subsequence(syn,text_tokens):
                    if category['name'] in self.special_categories and \
                        self.remove_special(category['name'],syn[0],text_tokens,text):
                            # syn[0] as all special categories are one-word.
                            break
                    assigned_categories.append(category)
                    break

        return assigned_categories


def split_data(cfg):
    category_split = io.load_json_object(cfg.coco_category_split)
    seen_categories = []
    if cfg.dataset_name in ['vqa','coco_captions']:
        seen_category_groups = ['shared','held_from_det']
        unseen_category_groups = ['held_from_vqa']
    elif cfg.dataset_name in ['coco_detection','coco_classification']:
        seen_category_groups = ['shared','held_from_vqa']
        unseen_category_groups = ['held_from_det']
    else:
        raise NotImplementedError

    for category_split_name in seen_category_groups:
        seen_categories.extend(category_split[category_split_name])
    
    unseen_categories = []
    for category_split_name in unseen_category_groups:
        unseen_categories.extend(category_split[category_split_name])

    images = io.load_json_object(cfg.split_coco_images)
    original_dataset = io.load_json_object(
        os.path.join(cfg.original_split,cfg.dataset[cfg.subset]))
    original_dataset_filtered_by_image = []
    for sample in tqdm(original_dataset):
        if sample['image']['subset']==images['subset'] \
            and sample['image']['image_id'] in images['image_ids']:
            original_dataset_filtered_by_image.append(sample)
    
    print('Original dataset:',len(original_dataset))
    print('Filtered by image split:',len(original_dataset_filtered_by_image))
    
    seen_category_asssigner = AssignCocoCategories(seen_categories,SYNONYMS)
    unseen_category_asssigner = AssignCocoCategories(unseen_categories,SYNONYMS)
    new_dataset = []
    discarded_dataset = []
    for i,sample in enumerate(tqdm(original_dataset_filtered_by_image)):
        assigned_categories = {
            'seen': [c['name'] for c in seen_category_asssigner.assign(sample)],
            'unseen': [c['name'] for c in unseen_category_asssigner.assign(sample)]
        }
        sample['coco_categories'] = assigned_categories
        if cfg.subset in ['train','val']:
            if len(assigned_categories['unseen'])==0 and \
                sample['image']['subset']==images['subset'] and \
                sample['image']['image_id'] in images['image_ids']:
                new_dataset.append(sample)
            else:
                discarded_dataset.append(sample)
        else:
            new_dataset.append(sample)

        if i%1000==0:
            tqdm.write('Selected: {}, Discarded: {}'.format(
                len(new_dataset),len(discarded_dataset)))
        

    io.dump_json_object(
        new_dataset,
        os.path.join(cfg.exp_dir,f'{cfg.subset}.json'))
    
    unseen_category_count = Counter()
    seen_category_count = Counter()
    for sample in tqdm(new_dataset):
        for category_name in sample['coco_categories']['unseen']:
            unseen_category_count[category_name] += 1
        
        for category_name in sample['coco_categories']['seen']:
            seen_category_count[category_name] += 1
    
    category_counts = {
        'seen': seen_category_count,
        'unseen': unseen_category_count
    }
    io.dump_json_object(
        category_counts,
        os.path.join(cfg.exp_dir,f'{cfg.subset}_category_counts.json'))

    print('Original Dataset:',len(original_dataset_filtered_by_image))
    print('Dataset:',len(new_dataset))


def compute_stats(cfg):
    images = io.load_json_object(cfg.split_coco_images)
    original_dataset = io.load_json_object(
        os.path.join(cfg.original_split,cfg.dataset[cfg.subset]))
    original_dataset_filtered_by_image = []
    for sample in tqdm(original_dataset):
        if sample['image']['subset']==images['subset'] \
            and sample['image']['image_id'] in images['image_ids']:
            original_dataset_filtered_by_image.append(sample)
    print('Original Dataset:',len(original_dataset_filtered_by_image))
    dataset = io.load_json_object(os.path.join(cfg.exp_dir,f'{cfg.subset}.json'))
    print('Dataset:',len(dataset))
    if cfg.subset=='test':
        test_seen = 0
        test_unseen = 0
        for sample in dataset:
            if len(sample['coco_categories']['unseen'])>0:
                test_unseen+=1
            else:
                test_seen+=1
        
        print('Seen Test:',test_seen)
        print('Unseen Test:',test_unseen)


@hydra.main(config_path='../configs',config_name='data/split_data_by_categories.yaml')
def main(cfg):
    print(cfg.pretty())
    if cfg.stats_only is True:
        compute_stats(cfg)
    else:
        split_data(cfg)
        

if __name__=='__main__':
    main()
