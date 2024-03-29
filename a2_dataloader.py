import locale
import os
import re
from string import punctuation
from collections import Counter
from typing import Optional, Union, Tuple, Type, Sequence, IO
import gzip
import torch


TOKENIZER_PATTERN = re.compile(r'[' + re.escape(punctuation) + r'\d\s]+')

locale.setlocale(locale.LC_ALL, 'C')  # ensure reproducible sorting

__all__ = [
    'get_dir_lines',
    'build_vocab_from_dir',
    'word2id_to_id2word',
    'id2word_to_word2id',
    'write_word2id_to_file',
    'read_word2id_from_file',
    'get_common_prefixes',
    'wmt16Dataset',
    'wmt16DataLoader',
]


def get_dir_lines(dir_: str, lang: str, filenames: Sequence[str] = None) -> None:
    '''Generate line info from data in a directory for a given language

    Parameters
    ----------
    dir_ : str
        A path to the transcription directory.
    lang : {'e', 'f'}
        Whether to tokenize the English sentences ('e') or Turkish ('f').
    filenames : sequence, optional
        Only tokenize sentences with matching names. If :obj:`None`, searches
        the whole directory in C-sorted order.

    Yields
    ------
    tokenized, filename, offs : list
        `tokenized` is a list of tokens for a line. `filename` is the source
        file. `offs` is the start of the sentence in the file, to seek to.
        Lines are yielded by iterating over lines in each file in the order
        presented in `filenames`.
    '''
    _in_set_check('lang', lang, {'e', 'f'})
    lang = '.' + lang
    if filenames is None:
        filenames = sorted(os.listdir(dir_))
    for filename in filenames:
        if filename.endswith(lang):
            with open(os.path.join(dir_, filename), encoding='utf-8') as f:
                offs = f.tell()
                line = f.readline()
                while line:
                    yield [
                        w for w in TOKENIZER_PATTERN.split(line.lower()) if w
                    ], filename, offs
                    offs = f.tell()
                    line = f.readline()


def build_vocab_from_dir(
        train_dir_: str,
        lang: str,
        max_vocab: int = 5000) -> dict:
    '''Build a vocabulary (words->ids) from transcriptions in a directory

    Parameters
    ----------
    train_dir_ : str
        A path to the transcription directory. ALWAYS use the training
        directory, not the test, directory, when building a vocabulary.
    lang : {'e', 'f'}
        Whether to build the English vocabulary ('e') or the Turkish one ('f').
    max_vocab : int, optional
        The size of your vocabulary. Words with the greatest count will be
        retained.

    Returns
    -------
    word2id : dict
        A dictionary of keys being words, values being ids. There will be an
        entry for each id between ``[0, max_vocab - 1]`` inclusive.
    '''
    _in_range_check('max_vocab', max_vocab, 3)
    word2count = Counter()
    for tokenized, _, _ in get_dir_lines(train_dir_, lang):
        word2count.update(tokenized)
    word2count = sorted(
        word2count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    word2count = word2count[:max_vocab - 3]
    return dict((v[0], i) for i, v in enumerate(word2count))


def word2id_to_id2word(word2id: dict) -> dict:
    '''word2id -> id2word'''
    return dict((v, k) for (k, v) in word2id.items())


def id2word_to_word2id(id2word: dict) -> dict:
    '''id2word -> word2id'''
    return dict((v, k) for (k, v) in id2word.items())


def write_word2id_to_file(word2id: dict, file_: Union[str, IO]) -> None:
    '''Write word2id map to a file

    Parameters
    ----------
    word2id : dict
        A dictionary of keys being words, values being ids
    file_ : str or file
        A file to write `word2id` to. If a path that ends with ``.gz``, it will
        be gzipped.
    '''
    if isinstance(file_, str):
        if file_.endswith('.gz'):
            with gzip.open(file_, mode='wt') as file_:
                return write_word2id_to_file(word2id, file_)
        else:
            with open(file_, 'w') as file_:
                return write_word2id_to_file(word2id, file_)
    id2word = word2id_to_id2word(word2id)
    for i in range(len(id2word)):
        file_.write('{} {}\n'.format(id2word[i], i))


def read_word2id_from_file(file_: Union[str, IO]) -> dict:
    '''Read word2id map from a file

    Parameters
    ----------
    file_ : str or file
        A file to read `word2id` from. If a path that ends with ``.gz``, it
        will be de-compressed via gzip.

    Returns
    -------
    word2id : dict
        A dictionary of keys being words, values being ids
    '''
    if isinstance(file_, str):
        if file_.endswith('.gz'):
            with gzip.open(file_, mode='rt') as file_:
                return read_word2id_from_file(file_)
        else:
            with open(file_) as file_:
                return read_word2id_from_file(file_)
    ids = set()
    word2id = dict()
    for line in file_:
        line = line.strip()
        if not line:
            continue
        word, id_ = line.split()
        id_ = int(id_)
        if id_ in ids:
            raise ValueError(f'Duplicate id {id_}')
        if word in word2id:
            raise ValueError(f'Duplicate word {word}')
        ids.add(id_)
        word2id[word] = id_
    _word2id_validity_check('word2id', word2id)
    return word2id


def get_common_prefixes(dir_: str) -> Sequence[str]:
    '''Return a list of file name prefixes common to both English and Turkish

    A prefix is common to both English and Turkish if the files
    ``<dir_>/<prefix>.e`` and ``<dir_>/<prefix>.f`` both exist.

    Parameters
    ----------
    dir_ : str
        A path to the transcription directory.

    Returns
    -------
    common : list
        A C-sorted list of common prefixes
    '''
    all_fns = os.listdir(dir_)
    english_fns = set(fn[:-2] for fn in all_fns if fn.endswith('.e'))
    turkish_fns = set(fn[:-2] for fn in all_fns if fn.endswith('.f'))
    del all_fns
    common = english_fns & turkish_fns
    if not common:
        raise ValueError(
            f'Directory {dir_} contains no common files ending in .e or '
            f'.f. Are you sure this is the right directory?')
    return sorted(common)


class wmt16Dataset(torch.utils.data.Dataset):
    '''A dataset of a partition of the Mukayese

    Indexes bitext sentence pairs ``F, E``, where ``F`` is the source language
    sequence and ``E`` is the corresponding target language sequence.

    Parameters
    ----------
    dir_ : str
        A path to the data directory
    turkish_word2id : dict or str
        Either a dictionary of Turkish words to ids, or a path pointing to one.
    english_word2id : dict or str
        Either a dictionary of English words to ids, or a path pointing to one.
    source_language : {'e', 'f'}, optional
        Specify the language we're translating from. By default, it's Turkish
        ('f'). In the case of English ('e'), ``F`` is still the source language
        sequence, but it now refers to English.
    prefixes : sequence, optional
        A list of file prefixes in `dir_` to consider part of the dataset. If
        :obj:`None`, will search for all common prefixes in the directory.

    Attributes
    ----------
    dir_ : str
    source_language : {'e', 'f'}
    source_unk : int
        A special id to indicate a source token was out-of-vocabulary.
    source_pad_id : int
        A special id used for right-padding source-sequences during batching
    source_vocab_size : int
        The total number of unique ids in source sequences. All ids are bound
        between ``[0, source_vocab_size - 1]`` inclusive. Includes
        `source_unk` and `source_pad_id`.
    target_unk : int
        A special id to indicate a target token was in-vocabulary.
    target_sos : int
        A special id to indicate the start of a target sequence. One SOS token
        is prepended to each target sequence ``E``.
    target_eos : int
        A special id to indicate the end of a target sequence. One EOS token
        is appended to each target sequence ``E``.
    target_vocab_size : int
        The total number of unique ids in target sequences. All ids are bound
        between ``[0, target_vocab_size - 1]`` inclusive. Includes
        `target_unk`, `target_sos`, and `target_eos`.
    pairs : tuple
    '''

    def __init__(
            self, dir_: str,
            turkish_word2id: Union[dict, str],
            english_word2id: Union[dict, str],
            source_language: str = 'f',
            prefixes: Sequence[str] = None):
        _in_set_check('source_language', source_language, {'e', 'f'})
        if isinstance(turkish_word2id, str):
            turkish_word2id = read_word2id_from_file(turkish_word2id)
        else:
            _word2id_validity_check('turkish_word2id', turkish_word2id)
        if isinstance(english_word2id, str):
            english_word2id = read_word2id_from_file(english_word2id)
        else:
            _word2id_validity_check('english_word2id', english_word2id)
        if prefixes is None:
            prefixes = get_common_prefixes(dir_)
        english_fns = (p + '.e' for p in prefixes)
        turkish_fns = (p + '.f' for p in prefixes)
        english_l = get_dir_lines(dir_, 'e', english_fns)
        turkish_l = get_dir_lines(dir_, 'f', turkish_fns)
        if source_language == 'f':
            source_word2id = turkish_word2id
            target_word2id = english_word2id
        else:
            source_word2id = english_word2id
            target_word2id = turkish_word2id
        pairs = []
        F_unk, F_pad = range(len(source_word2id), len(source_word2id) + 2)
        E_unk, E_sos, E_eos = range(
            len(target_word2id), len(target_word2id) + 3)
        for (e, e_fn, _), (f, f_fn, _) in zip(english_l, turkish_l):
            assert e_fn[:-2] == f_fn[:-2]
            if not e or not f:
                print(e_fn[:-2])
                assert not e and not f  # if either is empty, both should be
                continue
            if source_language == 'f':
                F, E = f, e
            else:
                F, E = e, f
            F = torch.tensor([source_word2id.get(w, F_unk) for w in F])
            E = torch.tensor(
                [E_sos] + [target_word2id.get(w, E_unk) for w in E] + [E_eos])
            if torch.all(F == F_unk) and torch.all(E[1:-1] == E_unk):
                # skip sentences that are solely OOV
                continue
            pairs.append((F, E))
        self.dir_ = dir_
        self.source_language = source_language
        self.source_vocab_size = len(source_word2id) + 2  # pad id and unk
        self.source_unk = F_unk
        self.source_pad_id = F_pad
        self.target_unk = E_unk
        self.target_sos = E_sos
        self.target_eos = E_eos
        self.target_vocab_size = len(target_word2id) + 3  # unk, sos, and eos
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[str, str]:
        return self.pairs[i]


class wmt16DataLoader(torch.utils.data.DataLoader):
    '''A DataLoader yielding batches of bitext

    Consult :class:`wmt16Dataset` for a description of parameters and
    attributes

    Parameters
    ----------
    dir_ : str
    turkish_word2id : dict or str
    english_word2id : dict or str
    source_language : {'e', 'f'}, optional
    prefixes : sequence, optional
    kwargs : optional
        See :class:`torch.utils.data.DataLoader` for additional arguments.
        Do not specify `collate_fn`.
    '''

    def __init__(
            self, dir_: str,
            turkish_word2id: Union[dict, str],
            english_word2id: Union[dict, str],
            source_language: str = 'f',
            prefixes: Sequence[str] = None, **kwargs):
        if 'collate_fn' in kwargs:
            raise TypeError(
                "wmt16DataLoader() got an unexpected keyword argument "
                "'collate_fn'")
        dataset = wmt16Dataset(
            dir_, turkish_word2id, english_word2id, source_language, prefixes)
        super().__init__(dataset, collate_fn=self.collate, **kwargs)

    def collate(self, seq):
        F, E = zip(*seq)
        F_lens = torch.tensor([len(f) for f in F])
        F = torch.nn.utils.rnn.pad_sequence(
            F, padding_value=self.dataset.source_pad_id)
        E = torch.nn.utils.rnn.pad_sequence(
            E, padding_value=self.dataset.target_eos)
        return F, F_lens, E


def _in_range_check(
        name: str, value: int,
        low: Union[int, float] = -float('inf'),
        high: Union[int, float] = float('inf'),
        error: Exception = Type[ValueError]):
    if value < low:
        raise error(f'{name} ({value}) is less than {low}')
    if value > high:
        raise error(f'{name} ({value}) is greater than {high}')


def _in_set_check(name: str, value: int, set_: str,
        error: Type[Exception] = ValueError):
    if value not in set_:
        raise error(f'{name} not in {set_}')


def _word2id_validity_check(name: str, word2id: dict,
        error: Type[Exception] = ValueError):
    if set(word2id.values()) != set(range(len(word2id))):
        raise error(
            f'Ids in {name} should be contiguous and span [0, len({name}) - 1]'
            f' inclusive')
