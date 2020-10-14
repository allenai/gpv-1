import hydra
import numpy as np

from .models.bert import Bert
import utils.io as io


def embed_vocab(vocab,vocab_embed_npy,batch_size):
    V = len(vocab)
    B = batch_size

    bert = Bert(None).cuda()
    embed = []

    R = V//B
    if V!=R*B:
        R = R+1

    for i in range(R):
        words = vocab[i*B:min((i+1)*B,V)]
        embed.append(bert(words)[0][:,0].detach().cpu().numpy())

    if len(embed) > 1:
        embed = np.concatenate(embed,0)
    else:
        embed = embed[0]

    print('Vocab embed dim:',embed.shape)
    np.save(vocab_embed_npy,embed)


@hydra.main(config_path=f'../../configs',config_name=f"exp/gpv_box_text")
def main(cfg):
    vocab = io.load_json_object(cfg.model.vocab)
    vocab_embed_npy = cfg.model.vocab_embed
    embed_vocab(vocab,vocab_embed_npy,10)

if __name__=='__main__':
    main()

