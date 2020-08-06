from src.dictionary import Dictionary
import numpy as np
import io

def read_txt_embeddings(lang,full_vocab=False):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    emb_path = 'vectors/wiki.{}.vec'.format(lang)
    _emb_dim_file = 300
    max_vocab = 200000

    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        print("Word '%s' found twice in embedding file" % (word))
                        pass
                else:
                    if not vect.shape == (_emb_dim_file,):
                        print("Invalid dimension (%i) for word '%s' in line %i." % (vect.shape[0], word, i))
                        continue
                assert vect.shape == (_emb_dim_file,), i
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    # logger.info("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)

    assert embeddings.shape == (len(dico), _emb_dim_file)
    return dico, embeddings

if __name__=='__main__':
    languages = ['English','Spanish','Italian']
    langs = ['en','es','it']
    title = 'humanRights'

    for lang,language in zip(langs,languages):
        dico,embeddings = read_txt_embeddings(lang)
        with io.open('{}/texts/{}'.format(title,language), 'r', encoding='utf-8') as f:
            embs,words = [],[]
            article = False
            sentence_num = 1
            for line in f:
                space_split = line.split(' ')
                if space_split[0] in ['Articolo','Article','Artículo']:
                    article=True
                    print(line)
                    continue
                if article==False:
                    continue
                for s in space_split:
                    s.lower()
                    s = s.replace('\n','')
                    if s=='':
                        continue
                    if s[-1] not in ',.':
                        if s in dico.word2id:
                            emb = embeddings[dico.word2id[s]]
                        else:
                            emb = np.zeros(300)
                        embs.append(emb)
                        words.append(s)
                    else:
                        if s[:-1] in dico.word2id:
                            emb = embeddings[dico.word2id[s[:-1]]]
                        else:
                            emb = np.zeros(300)
                        embs.append(emb)
                        words.append(s[:-1])
                        embs.append(embeddings[dico.word2id[s[-1]]])
                        words.append(s[-1])
                        if s[-1] in ',.':
                            embs_np = np.array(embs)
                            print(sentence_num,words)
                            np.save('{}/vectorized_texts/comma/{}/comma{}'.format(title,lang,sentence_num), embs_np)
                            embs,words = [],[]
                            sentence_num += 1