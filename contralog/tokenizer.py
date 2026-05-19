from tokenizers import (
    ByteLevelBPETokenizer,
    SentencePieceBPETokenizer,
    Tokenizer,
    normalizers,
    pre_tokenizers,
    models,
    trainers,
)
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers.implementations import (
    BertWordPieceTokenizer,
    SentencePieceUnigramTokenizer,
)

def make_raw_bpe(vocab_size:int, fit_messages:iter):
    fit_messages = list(fit_messages)
    tokenizer  = Tokenizer(BPE())
    tokenizer.model.unk_token = '[UNK]'
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", split=False,
                                    prepend_scheme='always')
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.NFKC()])
    trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=['[UNK]'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(fit_messages, trainer=trainer)
    return tokenizer

def make_tokenizer(type:str, vocab_size:int, fit_messages:iter):
    #--- Word
    if type == "word":
        tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=['[UNK]']
        )

        tokenizer.train_from_iterator(fit_messages, trainer=trainer)

    elif type == "word_digits":
        tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                    pre_tokenizers.Whitespace(), pre_tokenizers.Digits()])

        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=['[UNK]']
        )

        tokenizer.train_from_iterator(fit_messages, trainer=trainer)

    #--- WordPiece
    elif type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token='[UNK]', max_input_chars_per_word=1_000))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = WordPieceTrainer(
                    vocab_size=vocab_size,
                    min_frequency=2,
                    continuing_subword_prefix="##",
                    special_tokens=['[UNK]'])
        tokenizer.train_from_iterator(fit_messages, trainer=trainer)

    #--- BPE
    elif type =="bpe":
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tokenizer._tokenizer.model.unk_token = '[UNK]'
        tokenizer.train_from_iterator(fit_messages, vocab_size=vocab_size, special_tokens=['[UNK]'])
        
    #--- Sentence
    elif type == "sentence":
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        
        tokenizer._tokenizer.model.unk_token = '[UNK]'
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", split=True, prepend_scheme='always')
        tokenizer.train_from_iterator(fit_messages, vocab_size=vocab_size, special_tokens=['[UNK]'], 
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2)

        
    #--- Uni - also sentence (but with unigram not bpe)
    elif type == "uni":
        tokenizer = SentencePieceUnigramTokenizer(
            replacement="▁",
            add_prefix_space=True,
        )
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tokenizer.train_from_iterator(fit_messages, vocab_size=vocab_size, unk_token="[UNK]")
    
    #--- BERT
    elif type == "bert":
        assert False, "TODO add path to vocab file for BertWordPieceTokenizer"
        tokenizer = BertWordPieceTokenizer("TODO add path here",  # You need to download this file
        lowercase=True)
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    
    #--- Raw BPE
    elif type == "raw_bpe":
        tokenizer = make_raw_bpe(vocab_size, fit_messages)

    #--- Character-level
    elif type == "char":
        tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")

        trainer = trainers.WordLevelTrainer(
            vocab_size=257,#vocab_size,
            min_frequency=2,
            special_tokens=['[UNK]']
        )

        tokenizer.train_from_iterator(fit_messages, trainer=trainer)

    else:
        raise ValueError(f"Unknown tokenizer type: {type}")
    return tokenizer