import pandas as pd
import numpy as np
import re
from random import random
import emoji
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from transformers import RobertaTokenizer, DebertaTokenizer
from nltk.tokenize import TweetTokenizer
from cleaner import *


apostrophes = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
more_apostrophes = {'cannot': "can not", 'Gonna': "go to", 'gonna': "go to", 'tryna': "try to", 'wanna': "want to", 'coronavirus': "virus", 'wanted': "want", 'weeks': "week", 'feeling': "feel", 'says': "say", 'saying': "say", 'says': "say", 'GIF': "gif", 'waiting': "wait", 'Covid': "virus", 'hugs': "hug", 'gave': "give", 'COVID19': "virus", 'installing': "install", 'wants': "want", 'knows': "know", 'describes': "describe", 'asked': "ask", 'finally': "final", 'minutes': "minute", 'died': "die", 'tired': "tire", 'quickly': "quick", 'gotta': "go to", 'deaths': "death", 'means': "mean", 'took': "take", 'feels': "feel", 'fans': "fan", 'numbers': "number", 'lives': "live", 'safely': "safe", 'tried': "try", 'businesses': "business", '1st': "first", '2nd': "second", 'decided': "decide", '3rd': "third", 'hates': "hate", 'dont': "do not", 'lonely': "lone", 'totally': "total", 'BREAKING': "break", 'gifs': "gif", 'goes': "go", 'thoughts': "thought", 'campaigning': "campaign", 'immediately': "immediate", 'teammates': "team mate", 'knew': "know", 'politicians': "politician", 'distancing': "distance", 'reopening': "reopen", 'pls': "please", 'AGAIN': "again", 'tears': "tear", 'supposed': "suppose", 'loved': "love", 'ppl': "people", 'drinking': "drink", 'Guidelines': "guide line", 'losing': "lose", 'Conference': "conference", 'officially': "official", 'OPENING': "open", 'buying': "buy", 'Gif': "gif", 'looks': "look", 'bought': "buy", 'likes': "like", 'truely': "true", 'happened': "happen", 'putting': "put", 'families': "family", 'moved': "move", 'Raise': "raise", 'helped': "help", 'vibes': "vibe", 'voting': "vote", 'showed': "show", 'Instagram': "instagram", 'spent': "spend", 'watched': "watch", 'kinda': "kind of", 'Governor': "governor", 'Coronavirus': "virus", 'lmao': "laugh", 'seems': "seem", 'staying': "stay", 'listening': "listen", 'accounts': "account", 'GIVE': "give", 'gimme': "give me", 'winking': "wink", 'shrugging': "shrug", 'facepalming': "face palm", 'LIBERATE': "liberate", 'DMs': "direct mail", 'idk': "I do not know", 'Idk': "I do not know", 'cuz': "because", 'yall': "you all", 'FOLLOW': "follow", 'TESTING': "testing", 'wtf': "what the fuck", 'FUCKING': "fucking", 'Cryin': "crying", '7th': "seventh", '4TH': "fourth", 'VACATION': "vacation", 'pandemic': "disease", 'covid': "virus", 'COVID': "virus", 'CoronaVirus': "virus", 'virus19': "virus", 'ScottyFromMarketing': "Scotty from marketing", 'Frequently': "frequently", 'corona': "virus", 'Polling': "polling", '4th': "fourth", 'smirking': "amused", 'smh': "shake my head", 'HUMANS': "humans", 'POTUS': "President", 'Asking': "asking", 'omg': "oh my god", 'tbh': "to be honest", 'NOVEMBER': "November", 'bitches': "bitch", 'HUNGRY': "hungry", 'Shout': "shout", 'btw': "by the way", 'Couldn': "Could not", 'Gives': "gives", 'Sleepy': "sleepy", 'STAY': "stay", 'WTF': "what the fuck", 'EVERYONE': "everyone", 'GOING': "going", 'Candidates': "candidates"}
tokenizer = TweetTokenizer()


def get_vocab(corpus):
    vocabulary = Counter()
    for sentance in corpus:
        for word in sentance.split():
            vocabulary.update([word])
    return vocabulary


def check_coverage(vocabs, roberta_vocab):
    known_words = {}
    unknown_words = {}
    known_count = 0
    unknown_count = 0
    for word in tqdm(vocabs.keys(), desc='Checking: '):
        if word in list(roberta_vocab.keys()):
            known_words[word] = roberta_vocab[word]
            known_count += vocabs[word]
        elif 'Ġ' + word in list(roberta_vocab.keys()):
            # If we have deep look in roberta tokenizer, many words combine with 'Ġ' since roberta is byte pair encoding
            known_words[word] = roberta_vocab['Ġ' + word]
            known_count += vocabs[word]            
        else:
            unknown_words[word] = vocabs[word]
            unknown_count += vocabs[word]
    print("Found embeddings for {:.3%} ({} / {}) of vocab".format(len(known_words) / len(vocabs), len(known_words), len(vocabs)))
    print("Found embeddings for {:.3%} ({} / {}) of all text".format(known_count / (known_count + unknown_count), known_count, (known_count + unknown_count)))
    return unknown_words


def print_coverage(df_text, df_reply, roberta_vocab, mode):
    text_vocab = get_vocab(df_text.values)
    reply_vocab = get_vocab(df_reply.values)
    if mode == 'train':
        print("train text unique vocab count is: {}".format(len(text_vocab)))
        print("train reply unique vocab count is: {}".format(len(reply_vocab)))
    elif mode == 'dev':
        print("dev text unique vocab count is: {}".format(len(text_vocab)))
        print("dev reply unique vocab count is: {}".format(len(reply_vocab)))
    else:
        print("test text unique vocab count is: {}".format(len(text_vocab)))
        print("test reply unique vocab count is: {}".format(len(reply_vocab)))
    unknown_text = check_coverage(text_vocab, roberta_vocab)
    print()
    unknown_reply = check_coverage(reply_vocab, roberta_vocab)


def clean_weird(text):
    specials = ["’", "‘", "´", "`"]
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("´", "'")
    text = text.replace("`", "'")
    return text


def change_apostrophes(text):
    # Replace apostrophes to original term
    for key in apostrophes.keys():
        text = text.replace(key, apostrophes[key])
    return text


def change_punc(text):
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])
    for p in punct:
        text = text.replace(p, ' {} '.format(p))
    return text


def distinct_emoji_lis(string):
    """Resturns distinct list of emojis from the string"""
    distinct_list = list({c for c in string if c in emoji.unicode_codes.UNICODE_EMOJI})
    return distinct_list


def change_emoji_to_text(text):
    """
    Input: text
    Output: demojize text
    """
    ori_text = text
    distinct_emoji = distinct_emoji_lis(text)
    for each_emoji in distinct_emoji:
        first_appear = ori_text.index(each_emoji)
        new_text = ''
        for tid, token in enumerate(ori_text):
            if token == each_emoji and tid != first_appear:
                new_text += ''
            else:
                new_text += token
        ori_text = new_text
    ori_text = emoji.demojize(ori_text).replace(':', ' ').replace('_', ' ')
    return ori_text


def change_punc(text):
    for key in more_apostrophes.keys():
        text = text.replace(key, more_apostrophes[key])
    return text


def map_special(text):
    tokens = tokenizer.tokenize(text)
    new_text = []
    for token in tokens:
        lowercased_token = token.lower()
        if token.startswith("@"):
            new_text.append((token, '@USER'))
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            new_text.append((token, 'HTTPURL'))

    for original_token, replaced_token in new_text:
        text = text.replace(original_token, replaced_token)
    return text


def preprocessing():
    df_train = read_json('./original_data/new_train.json')
    df_dev = read_json('./original_data/new_dev.json')
    df_test = read_json('./original_data/new_eval.json')

    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    vocab = pd.read_json('deberta_vocab/vocab.json', typ='series')

    print("\n[Original coverage]")
    # print_coverage(df_train['text'], df_train['reply'], vocab, 'train')
    # print_coverage(df_dev['text'], df_dev['reply'], vocab, 'dev')
    # print_coverage(df_test['text'], df_test['reply'], vocab, 'test')

    # map user and url
    df_train['text'] = df_train.text.apply(map_special)
    df_train['reply'] = df_train.reply.apply(map_special)
    df_dev['text'] = df_dev.text.apply(map_special)
    df_dev['reply'] = df_dev.reply.apply(map_special)
    df_test['text'] = df_test.text.apply(map_special)
    df_test['reply'] = df_test.reply.apply(map_special)

    print("\n[Mapping users and urls]")
    # print_coverage(df_train['text'], df_train['reply'], vocab, 'train')
    # print_coverage(df_dev['text'], df_dev['reply'], vocab, 'dev')
    # print_coverage(df_test['text'], df_test['reply'], vocab, 'test')

    # clean weird punctuations
    df_train['text'] = df_train.text.apply(clean_weird)
    df_train['reply'] = df_train.reply.apply(clean_weird)
    df_dev['text'] = df_dev.text.apply(clean_weird)
    df_dev['reply'] = df_dev.reply.apply(clean_weird)
    df_test['text'] = df_test.text.apply(clean_weird)
    df_test['reply'] = df_test.reply.apply(clean_weird)

    print("\n[Cleaning weird coverage]")
    # print_coverage(df_train['text'], df_train['reply'], vocab, 'train')
    # print_coverage(df_dev['text'], df_dev['reply'], vocab, 'dev')
    # print_coverage(df_test['text'], df_test['reply'], vocab, 'test')

    # transform apostrophes
    df_train['text'] = df_train.text.apply(change_apostrophes)
    df_train['reply'] = df_train.reply.apply(change_apostrophes)
    df_dev['text'] = df_dev.text.apply(change_apostrophes)
    df_dev['reply'] = df_dev.reply.apply(change_apostrophes)
    df_test['text'] = df_test.text.apply(change_apostrophes)
    df_test['reply'] = df_test.reply.apply(change_apostrophes)

    print("\n[Transforming apostrophes coverage]")
    # print_coverage(df_train['text'], df_train['reply'], vocab, 'train')
    # print_coverage(df_dev['text'], df_dev['reply'], vocab, 'dev')
    # print_coverage(df_test['text'], df_test['reply'], vocab, 'test')

    # mapping unknown to known punctuations
    df_train['map_punc_text'] = df_train.text.apply(change_punc)
    df_train['map_punc_reply'] = df_train.reply.apply(change_punc)
    df_dev['map_punc_text'] = df_dev.text.apply(change_punc)
    df_dev['map_punc_reply'] = df_dev.reply.apply(change_punc)
    df_test['map_punc_text'] = df_test.text.apply(change_punc)
    df_test['map_punc_reply'] = df_test.reply.apply(change_punc)

    print("\n[Mapping unknown to known punctuations coverage]")
    # print_coverage(df_train['map_punc_text'], df_train['map_punc_reply'], vocab, 'train')
    # print_coverage(df_dev['map_punc_text'], df_dev['map_punc_reply'], vocab, 'dev')
    # print_coverage(df_test['map_punc_text'], df_test['map_punc_reply'], vocab, 'test')

    # try demojize to text and unique same emojis
    df_train['map_demojize_text'] = df_train.map_punc_text.apply(change_emoji_to_text)
    df_train['map_demojize_reply'] = df_train.map_punc_reply.apply(change_emoji_to_text)
    df_dev['map_demojize_text'] = df_dev.map_punc_text.apply(change_emoji_to_text)
    df_dev['map_demojize_reply'] = df_dev.map_punc_reply.apply(change_emoji_to_text)
    df_test['map_demojize_text'] = df_test.map_punc_text.apply(change_emoji_to_text)
    df_test['map_demojize_reply'] = df_test.map_punc_reply.apply(change_emoji_to_text)

    print("\n[Demojize to text and unique same emojis coverage]")
    # print_coverage(df_train['map_demojize_text'], df_train['map_demojize_reply'], vocab, 'train')
    # print_coverage(df_dev['map_demojize_text'], df_dev['map_demojize_reply'], vocab, 'dev')
    # print_coverage(df_test['map_demojize_text'], df_test['map_demojize_reply'], vocab, 'test')

    # transform more words
    df_train['map_more_punc_text'] = df_train.map_demojize_text.apply(change_punc)
    df_train['map_more_punc_reply'] = df_train.map_demojize_reply.apply(change_punc)
    df_dev['map_more_punc_text'] = df_dev.map_demojize_text.apply(change_punc)
    df_dev['map_more_punc_reply'] = df_dev.map_demojize_reply.apply(change_punc)
    df_test['map_more_punc_text'] = df_test.map_demojize_text.apply(change_punc)
    df_test['map_more_punc_reply'] = df_test.map_demojize_reply.apply(change_punc)

    print("\n[Transforming more words coverage]")
    # print_coverage(df_train['map_more_punc_text'], df_train['map_more_punc_reply'], vocab, 'train')
    # print_coverage(df_dev['map_more_punc_text'], df_dev['map_more_punc_reply'], vocab, 'dev')
    # print_coverage(df_test['map_more_punc_text'], df_test['map_more_punc_reply'], vocab, 'test')

    # output preprocessed to json
    df_preprocessed = df_train[['idx', 'map_more_punc_text', 'categories', 'context_idx', 'map_more_punc_reply', 'mp4', 'label']].copy()
    df_preprocessed.columns = ['idx', 'text', 'categories', 'context_idx', 'reply', 'mp4', 'label']
    df_preprocessed.to_json('./processed_data/preprocess_new_train.json', orient='records')
    df_preprocessed_dev = df_dev[['idx', 'map_more_punc_text', 'categories', 'context_idx', 'map_more_punc_reply', 'mp4']].copy()
    df_preprocessed_dev.columns = ['idx', 'text', 'categories', 'context_idx', 'reply', 'mp4']
    df_preprocessed_dev.to_json('./processed_data/preprocess_new_dev.json', orient='records')
    df_preprocessed_test = df_test[['idx', 'map_more_punc_text', 'categories', 'context_idx', 'map_more_punc_reply', 'mp4']].copy()
    df_preprocessed_test.columns = ['idx', 'text', 'categories', 'context_idx', 'reply', 'mp4']
    df_preprocessed_test.to_json('./processed_data/preprocess_new_eval.json', orient='records')

    print("\n[Done]")


if __name__ == '__main__':
    preprocessing()