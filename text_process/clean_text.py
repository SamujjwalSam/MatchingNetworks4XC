# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Cleans the input texts along with text labels.

__description__ :
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "06-05-2019"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Clean_Text

__variables__   :

__methods__     :
"""

import re,spacy
# from spacy.lang.en.stop_words import STOP_WORDS
import unicodedata
from unidecode import unidecode
from os.path import join,isfile
from collections import OrderedDict
from logger.logger import logger
from file_utils import File_Util
from config import configuration as config
from config import platform as plat
from config import username as user

spacy_en = spacy.load("en")
# spacy_en = spacy.load("en_core_web_sm")

""" Cleaning Procedure (in order)::
5. Remove texts like: "This section may require cleanup to meet Wikipedia's quality standards."
"""
headings = ['## [edit] ','### [edit] ','<IMG>',"[citation needed]",]
wiki_patterns = ("It has been suggested that Incremental reading be merged into this article or section. (Discuss)",
                 "This section may require cleanup to meet Wikipedia's quality standards.",
                 'This article includes a list of references or external links, but its sources remain unclear because it lacks inline citations. Please improve this article by introducing more precise citations where appropriate.')


class Clean_Text(object):
    """ Class related to cleaning the input texts along with text labels. """

    def __init__(self,dataset_name: str = config["data"]["dataset_name"],
                 data_dir: str = config["paths"]["dataset_dir"][plat][user]):
        """ Initializes the parts of cleaning to be done. """
        super(Clean_Text,self).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = join(data_dir,self.dataset_name)

    @staticmethod
    def dedup_data(Y: dict,dup_cat_map: dict):
        """
        Replaces category ids in Y if it's duplicate.

        :param Y:
        :param dup_cat_map: {old_cat_id : new_cat_id}
        """
        for k,v in Y.items():
            commons = set(v).intersection(set(dup_cat_map.keys()))
            if len(commons) > 0:
                for dup_key in commons:
                    dup_idx = v.index(dup_key)
                    v[dup_idx] = dup_cat_map[dup_key]
        return Y

    @staticmethod
    def split_docs(docs: dict,criteria = ' '):
        """
        Splits a dict of idx:documents based on [criteria].

        :param docs: idx:documents
        :param criteria:
        :return:
        """
        splited_docs = OrderedDict()
        for idx,doc in docs:
            splited_docs[idx] = doc.split(criteria)
        return splited_docs

    @staticmethod
    def make_trans_table(specials="""< >  * ? " / \\ : |""",replace=' '):
        """
        Makes a transition table to replace [specials] chars within [text] with [replace].

        :param specials:
        :param replace:
        :return:
        """
        trans_dict = {chars:replace for chars in specials}
        trans_table = str.maketrans(trans_dict)
        return trans_table

    def clean_categories(self,categories: dict,
                         specials=""" ? ! _ - @ < > # , . * ? ' { } [ ] ( ) $ % ^ ~ ` : ; " / \\ : |""",
                         replace=' '):
        """Cleans categories dict by removing any symbols and lower-casing and returns set of cleaned categories
        and the dict of duplicate categories.

        :param: categories: dict of cat:id
        :param: specials: list of characters to clean.
        :returns:
            category_cleaned_dict : contains categories which are unique after cleaning.
            dup_cat_map : Dict of new category id mapped to old category id. {old_cat_id : new_cat_id}
        """
        category_cleaned_dict = OrderedDict()
        dup_cat_map = OrderedDict()
        dup_cat_text_map = OrderedDict()
        trans_table = self.make_trans_table(specials=specials,replace=replace)
        for cat,cat_id in categories.items():
            cat_clean = unidecode(str(cat)).translate(trans_table).lower().strip()
            if cat_clean in category_cleaned_dict.keys():
                dup_cat_map[categories[cat]] = category_cleaned_dict[cat_clean]
                dup_cat_text_map[cat] = cat_clean
            else:
                category_cleaned_dict[cat_clean] = cat_id
        return category_cleaned_dict,dup_cat_map,dup_cat_text_map

    def clean_sentences_dict(self,sentences: dict,specials="""_-@*#'"/\\""",replace=' '):
        """Cleans sentences dict and returns dict of cleaned sentences.

        :param: sentences: dict of idx:label
        :returns:
            sents_cleaned_dict : contains cleaned sentences.
        """
        sents_cleaned_dict = OrderedDict()
        trans_table = self.make_trans_table(specials=specials,replace=replace)
        for idx,text in sentences.items():
            sents_cleaned_dict[idx] = unidecode(str(text)).translate(trans_table)
        return sents_cleaned_dict

    def clean_sentences(self,sentences: list,specials="""_-@*#'"/\\""",replace=' '):
        """Cleans sentences dict and returns dict of cleaned sentences.

        :param: sentences: dict of idx:label
        :returns:
            sents_cleaned_dict : contains cleaned sentences.
        """
        sents_cleaned_dict = []
        trans_table = self.make_trans_table(specials=specials,replace=replace)
        for text in sentences:
            sents_cleaned_dict.append(unidecode(str(text)).translate(trans_table))
        return sents_cleaned_dict

    @staticmethod
    def remove_wiki_first_lines(doc: str,num_lines=6):
        """
        Removes first [num_lines] lines from wikipedia texts.

        :param doc:
        :param num_lines:
        """
        doc = doc.split("\n")[num_lines:]
        doc = list(filter(None,doc))
        return doc

    @staticmethod
    def tokenizer_spacy(input_text: str):
        """ Document tokenizer using spacy.

        :param input_text:
        :return:
        """
        input_text = spacy_en(input_text)
        tokens = []
        for token in input_text:
            tokens.append(token.text)
        return tokens

    def calculate_idf(self,docs: list,subtract: int = 1) -> dict:
        """ Calculates idf scores for each token in the corpus.

        :param docs:
        :param subtract: Removes this value from idf scores. Sometimes needed to get better scores.
        :return: Dict of token to idf score.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        ## Ussing TfidfVectorizer with spacy tokenizer; same tokenizer should be used everywhere.
        vectorizer = TfidfVectorizer(decode_error='ignore',lowercase=False,smooth_idf=False,
                                     tokenizer=self.tokenizer_spacy)
        tfidf_matrix = vectorizer.fit_transform(docs)
        idf = vectorizer.idf_
        idf_dict = dict(zip(vectorizer.get_feature_names(),idf - subtract))  ## Subtract 1 from idf to get better scores

        if isfile(join(self.dataset_dir,self.dataset_name + "_idf_dict.json")):
            File_Util.load_json(filename=self.dataset_name + "_idf_dict",file_path=self.dataset_dir)
        else:
            File_Util.save_json(idf_dict,filename=self.dataset_name + "_idf_dict",file_path=self.dataset_dir)

        return idf_dict

    @staticmethod
    def remove_wiki_categories(doc: list):
        """ Removes category (and some irrelevant and repetitive) information from wiki pages.

        :param doc:
        """
        for i,sent in enumerate(reversed(doc)):  ## Looping from the end of list till first match
            if sent.lower().startswith("Categories:".lower()) or sent.lower().startswith("Category:".lower()):
                del doc[-1]  ## Match found; remove from end before breaking
                del doc[-1]
                del doc[-1]
                break
            del doc[-1]  ## Need to remove from end
        return doc

    @staticmethod
    def remove_wiki_contents(doc: list,start: str = '## Contents',end: str = '## [edit]') -> list:
        """ Removes the contents section from wikipedia pages as it overlaps among documents.

        :param start:
        :param end:
        :param doc:
        """
        start_idx = -1
        end_idx = -1
        content_flag = False
        for i,sent in enumerate(doc):
            if sent.startswith(start):  ## Starting index of content
                content_flag = True
                start_idx = i
                continue
            if sent.startswith(end) and content_flag:  ## Ending index of content
                end_idx = i
                break
        del doc[start_idx:end_idx]
        return doc

    @staticmethod
    def remove_wiki_headings(doc: list) -> list:
        """ Removes wikipedia headings like '### [edit] ', '## [edit] '

        :param doc:
        """
        for i,sent in enumerate(doc):  ## Looping from the end of list till first match
            for heading in headings:
                if sent.startswith(heading):
                    del doc[i]  ## Match found; remove end before breaking
        return doc

    def clean_doc(self,doc: list,pattern=re.compile(r"\[\d\]"),replace='',symbols=('_','-','=','@','<','>','*','{',
                  '}','[',']','(',')','$','%','^','~','`',':',"\"","\'",'\\','/','|','#','##','###','####','#####')) -> list:
        """ Cleans a list of str.

        :param doc:
        :param pattern:
        :param replace:
        :param start:
        :param end:
        :param symbols:
        :return:
        """
        doc_cleaned = []
        for i,sent in enumerate(doc):
            for heading in headings:
                if sent.startswith(heading) or not sent:
                    sent = ''
                    break

            if sent:
                sent = sent.strip()
                sent = re.sub(pattern=pattern,repl=replace,string=sent)
                sent = re.sub(r'[^\x00-\x7F]+',' ',sent)

                sent = sent.strip()
                tokens = self.tokenizer_spacy(sent)
                tokens,numbers = self.find_numbers(tokens)

                sent_new = []
                for token in tokens:
                    if token not in symbols:
                        sent_new.append(token)  ## Only add words which are not present in [symbols].
                doc_cleaned.append(" ".join(sent_new))
        return doc_cleaned

    @staticmethod
    def read_stopwords(so_filepath: str = '',so_filename: str = 'stopwords_en.txt',encoding: str = "iso-8859-1") -> list:
        """ Reads the stopwords list from file.

        :param so_filepath:
        :param so_filename:
        :param encoding:
        """
        from os.path import join,isfile

        so_list = []
        if isfile(join(so_filepath,so_filename)):
            with open(join(so_filepath,so_filename),encoding=encoding) as so_ptr:
                for s_word in so_ptr:
                    so_list.append(s_word.strip())
        else:
            raise Exception("File not found at: [{}]".format(join(so_filepath,so_filename)))

        return so_list

    @staticmethod
    def remove_symbols(doc: list,symbols=(
            '_','-','=','@','<','>','*','{','}','[',']','(',')','$','%','^','~','`',':',"\"","\'",'\\','#','##','###',
            '####','#####')):
        """ Replaces [symbols] from [doc] with [replace].

        :param doc:
        :param symbols:
        :return:
        """
        txt_new = []
        for i,sent in enumerate(doc):
            sent_new = []
            sent = sent.strip()  ## Stripping extra spaces at front and back.
            for token in sent.split():
                if token not in symbols:
                    sent_new.append(token)  ## Add words which are not present in [symbols] or [stopwords].
            txt_new.append(" ".join(sent_new))
        return txt_new

    @staticmethod
    def remove_symbols_trans(doc: str,symbols="""* , _ [ ] > < { } ( )""",replace=' '):
        """ Replaces [symbols] from [doc] with [replace].

        :param doc:
        :param symbols:
        :param replace:
        :return:
        """
        doc = unidecode(str(doc))
        trans_dict = {chars:replace for chars in symbols}
        trans_table = str.maketrans(trans_dict)
        return doc.translate(trans_table)

    @staticmethod
    def remove_nonascii(doc: list) -> list:
        """ Removes all non-ascii characters.

        :param doc:
        :return:
        """
        txt_ascii = []
        for sent in doc:
            sent = re.sub(r'[^\x00-\x7F]+',' ',sent)
            ## Alternatives::
            ## sent2 = ''.join([i if ord(i) < 128 else ' ' for i in sent])
            ## sent3 = unidecode(str(sent))
            ## sent4 = sent.encode('ascii', 'replace').decode()
            ## sent5 = sent.encode('ascii', 'ignore').decode()
            txt_ascii.append(sent)

        return txt_ascii

    def doc_unicode2ascii(self,doc: list) -> list:
        """ Converts a list of non-ascii str to ascii str based on unicode complient.

        :param doc:
        :return:
        """
        txt_ascii = []
        for sent in doc:
            sent2 = self.unicode2ascii(sent)
            txt_ascii.append(sent2)

        return txt_ascii

    @staticmethod
    def unicode2ascii(sent: str):
        """ Turn a Unicode string to plain ASCII. Thanks to http://stackoverflow.com/a/518232/2809427

        :param sent:
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD',sent)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def remove_patterns(doc: list,pattern=None,replace=''):
        """ Remove references from wiki pages.

        :param replace:
        :param pattern:
        :param doc:
        """
        if pattern is None:
            pattern = re.compile(r"\[\d\]")
        doc2 = []
        for sent in doc:
            sent = re.sub(pattern=pattern,repl=replace,string=sent)
            doc2.append(sent)
        return doc2

    @staticmethod
    def sents_split(doc: str):
        """ Splits the document into sentences and remove stopwords.

        :param doc:
        """
        doc_spacy = spacy_en(doc)
        sentences = list(doc_spacy.sents)
        return sentences

    @staticmethod
    def spacy_sents2string(doc: list):
        """ Converts a list of spacy span to concatenated string.

        We usually get this type from clean_text.py file.
        :param doc:
        """
        sents = str()
        for sent in doc:
            sents = sents + ' ' + sent.text
        return sents

    @staticmethod
    def find_label_occurrences(doc,label: str):
        """ Finds all label indices within document.

        :param doc:
        :param label:
        """
        label_idx = []
        logger.debug(label)
        ## re can not work with patterns having '+' or '*' in it, ignoring str with these characters.
        if '+' in label: return False
        if '*' in label: return False

        for m in re.finditer(label,doc):
            label_idx.append((m.start(),m.end()))
        if label_idx:
            return label_idx
        return False

    @staticmethod
    def find_label_occurrences_2(doc,labels: list):
        """ Finds all label indices within document.

        :param doc:
        :param labels:
        """
        labels_indices = {}
        for lbl in labels:
            for m in re.finditer(lbl,doc):
                if lbl not in labels_indices:
                    labels_indices[lbl] = []
                labels_indices[lbl].append((m.start(),m.end()))
        return labels_indices

    @staticmethod
    def filter_html_categories_reverse(txt: list):
        """Filters categories from html text."""
        category_lines_list = []
        category_lines = ""
        copy_flag = False
        del_line = True
        remove_first_chars = 12  ## Length of "Categories:", to be removed from line.

        ## Categories are written in multiple lines, need to read all lines (till "##### Views").
        for i,line in enumerate(reversed(txt)):  ## Looping reversed
            if line.lower().startswith("##### Views".lower()):
                copy_flag = True  ## Start coping "Categories:" as "##### Views" started.
                del_line = False
            if line.lower().startswith("Retrieved from".lower()) or line.lower().startswith(
                    "\"http://en.wikipedia.org".lower()):
                copy_flag = False
                del txt[-1]
            if copy_flag:
                category_lines = line + " " + category_lines
                category_lines_list.insert(0,line)
                del txt[-1]
            if line.lower().startswith("Categories:".lower()):
                break
            elif line.lower().startswith("Category:".lower()):
                remove_first_chars = 11
                break
            if del_line:
                del txt[-1]

        category_lines = category_lines[:-12]  ## To remove "##### Views "
        hid_cats = None
        if "Hidden categories:".lower() in category_lines.lower():  ## Process hidden categories
            category_lines,hid_cats = category_lines.split("Hidden categories:")
            hid_cats = (hid_cats.split(" | "))  ## Do not add hidden categories to categories.
            hid_cats = [cat.strip() for cat in hid_cats]
            hid_cats = list(filter(None,hid_cats))  ## Removing empty items.

        ## Filtering Categories
        category_lines = category_lines[remove_first_chars:].split(" | ")

        category_lines = [cat.strip() for cat in category_lines]
        category_lines = list(filter(None,category_lines))  ## Removing empty items.
        return txt,category_lines,hid_cats

    @staticmethod
    def remove_url(doc):
        """ Removes URls from str.

        :param doc:
        :return:
        """
        return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','',doc,flags=re.MULTILINE)  ## Removes URLs

    @staticmethod
    def case_folding(sent: list,all_caps: bool = False) -> list:
        """ Converts all text to lowercase except all capital letter words.

        :param sent:
        :param all_caps:
        :return:
        """
        for pos in range(len(sent)):
            if sent[pos].isupper():
                continue
            else:
                sent[pos] = sent[pos].lower()
        return sent

    def format_digits(self,doc: list):
        """ Replaces number from a str.

        :return: text, list of numbers
        Ex:
        '1230,1485': 8d
        '-2': 1d
        3.0 : 2f
        """
        doc2 = []
        for sent in doc:
            sent,numbers = self.find_numbers(sent)
            doc2.append(sent)

        return doc2

    @staticmethod
    def find_numbers(text: list,ignore_len: int = 4,replace=True):
        """ Finds and replaces numbers in list of str.

        :param ignore_len: Ignore numbers less than [ignore_len] digits.
        :param text: strings that contains digit and words
        :param replace: bool to decide if numbers need to be replaced.
        :return: text, list of numbers

        Ex:
        '1230,1485': 8d
        '-2': 1d
        3.0 : 2f
        """
        import re

        numexp = re.compile(r'(?:(?:\d+,?)+(?:\.?\d+)?)')
        numbers = numexp.findall(" ".join(text))
        numbers = [num for num in numbers if len(str(num)) > ignore_len]  ## Ignore numbers less than 4 digits.
        if replace:
            for num in numbers:
                try:
                    i = text.index(num)
                    if num.isdigit():
                        text[i] = str(len(num)) + "d"
                    else:
                        try:  ## Check if float
                            num = float(num)
                            text[i] = str(len(str(num)) - 1) + "f"
                        except ValueError as e:  ## if not float, process as int.
                            text[i] = str(len(num) - 1) + "d"
                except ValueError as e:  ## No numbers within text
                    pass
        return text,numbers

    @staticmethod
    def stemming(token,lemmatize=False):
        """ Stems tokens.

        :param token:
        :param lemmatize:
        :return:
        """
        if lemmatize:
            from nltk.stem import WordNetLemmatizer

            wnl = WordNetLemmatizer()
            return wnl.lemmatize(token)
        else:
            from nltk.stem import PorterStemmer

            ps = PorterStemmer()
            return ps.stem(token)

    @staticmethod
    def tokenizer_re(sent:str,lowercase=False,remove_emoticons=True):
        """ Tokenize a string.

        :param sent:
        :param lowercase:
        :param remove_emoticons:
        :return:
        """
        import re

        emoticons_str = r'''
            (?:
                [:=;] # Eyes
                [oO\-]? # Nose (optional)
                [D\)\]\(\]/\\OpP] # Mouth
            )'''
        regex_str = [
            emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f]['
            r'0-9a-f]))+',  # URLs
            r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
            r'(?:[\w_]+)',  # other words
            r'(?:\S)'  # anything else
        ]
        tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')',
                               re.VERBOSE | re.IGNORECASE)
        emoticon_re = re.compile(r'^' + emoticons_str + '$',
                                 re.VERBOSE | re.IGNORECASE)

        ## TODO: remove emoticons only (param: remove_emoticons).
        tokens = tokens_re.findall(str(sent))
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for
                      token in tokens]
        return tokens

    @staticmethod
    def remove_symbols(tweet,stopword=False,punct=False,specials=False):
        """ Removes symbols.

        :param tweet:
        :param stopword:
        :param punct:
        :param specials:
        :return:
        """
        if stopword:
            from nltk.corpus import stopwords

            stopword_list = stopwords.words('english') + ['rt','via','& amp',
                                                          '&amp','amp','mr']
            tweet = [term for term in tweet if term not in stopword_list]

        if punct:
            from string import punctuation

            tweet = [term for term in tweet if term not in list(punctuation)]

        if specials:
            trans_dict = {chars:' ' for chars in specials}
            trans_table = str.maketrans(trans_dict)
            tweet = tweet.translate(trans_table)

            for pos in range(len(tweet)):
                tweet[pos] = tweet[pos].replace("@","")
                tweet[pos] = tweet[pos].replace("#","")
                tweet[pos] = tweet[pos].replace("-"," ")
                tweet[pos] = tweet[pos].replace("&"," and ")
                tweet[pos] = tweet[pos].replace("$"," dollar ")
                tweet[pos] = tweet[pos].replace("  "," ")
        return tweet


def clean_wiki2(doc: list,num_lines: int = 6) -> str:
    """ Cleans the wikipedia documents.

    """
    cls = Clean_Text()
    doc = doc[num_lines:]
    doc = list(filter(None,doc))
    doc = cls.remove_wiki_contents(doc)
    doc = cls.remove_wiki_headings(doc)
    doc = cls.remove_nonascii(doc)  ## Remove all non-ascii characters
    doc = cls.remove_patterns(doc)
    # doc = cls.remove_patterns(doc, pattern=re.compile(r"\""))  ## Removes " with sentences
    doc = cls.remove_symbols(doc)
    # doc = cls.format_digits(doc)
    doc = cls.sents_split(" ".join(doc))
    doc = cls.spacy_sents2string(doc)
    doc = cls.remove_url(doc)

    return doc


def clean_wiki(doc: list,num_lines: int = 6) -> str:
    """ Cleans the wikipedia documents.

    """
    cls = Clean_Text()
    doc = doc[num_lines:]
    doc = list(filter(None,doc))
    doc = cls.remove_wiki_contents(doc)
    doc = cls.clean_doc(doc)
    doc = cls.sents_split(" ".join(doc))
    doc = cls.spacy_sents2string(doc)
    doc = cls.remove_url(doc)

    return doc


def clean_wiki_pages(docs):
    docs_cleaned = []
    for doc in docs:
        docs_cleaned.append(clean_wiki(doc))
    return docs_cleaned


def main() -> None:
    """ main module to start code """
    doc = """
# Encryption

### From Wikipedia, the free encyclopedia

Jump to: navigation, search

"Encrypt" redirects here. For the film, see Encrypt (film).

This article is about algorithms for encryption and decryption. For an
overview of cryptographic technology in general, see Cryptography.

In cryptography, encryption is the process of transforming information
(referred to as plaintext) using an algorithm (called cipher) to make it
unreadable to anyone except those possessing special knowledge, usually
referred to as a key. The result of the process is encrypted information (in
cryptography, referred to as ciphertext). In many contexts, the word
encryption also implicitly refers to the reverse process, decryption (e.g.
âsoftware for encryptionâ can typically also perform decryption), to make
the encrypted information readable again (i.e. to make it unencrypted).

Encryption has long been used by militaries and governments to facilitate
secret communication. Encryption is now commonly used in protecting
information within many kinds of civilian systems. For example, in 2007 the
U.S. government reported that 71% of companies surveyed utilized encryption
for some of their data in transit.[1] Encryption can be used to protect data
"at rest", such as files on computers and storage devices (e.g. USB flash
drives). In recent years there have been numerous reports of confidential data
such as customers' personal records being exposed through loss or theft of
laptops or backup drives. Encrypting such files at rest helps protect them
should physical security measures fail. Digital rights management systems
which prevent unauthorized use or reproduction of copyrighted material and
protect software against reverse engineering (see also copy protection)are
another somewhat different example of using encryption on data at rest.

Encryption is also used to protect data in transit, for example data being
transferred via networks (e.g. the Internet, e-commerce), mobile telephones,
wireless microphones, wireless intercom systems, Bluetooth devices and bank
automatic teller machines. There have also been numerous reports of data in
transit being intercepted in recent years [2]. Encrypting data in transit also
helps to secure it as it is often difficult to physically secure all access to
networks. Encryption, by itself, can protect the confidentiality of messages,
but other techniques are still needed to protect the integrity and
authenticity of a message; for example, verification of a message
authentication code (MAC) or a digital signature. Standards and cryptographic
software and hardware to perform encryption are widely available, but
successfully using encryption to ensure security may be a challenging problem.
A single slip-up in system design or execution can allow successful attacks.
Sometimes an adversary can obtain unencrypted information without directly
undoing the encryption. See, e.g., traffic analysis, TEMPEST, or Trojan horse.

One of the earliest public key encryption applications was called Pretty Good
Privacy (PGP), according to Paul Rubens. It was written in 1991 by Phil
Zimmermann and was bought by Network Associates in 1997 and is now called PGP
Corporation.

There are a number of reasons why an encryption product may not be suitable in
all cases. First e-mail must be digitally signed at the point it was created
to provide non-repudiation for some legal purposes, otherwise the sender could
argue that it was tampered with after it left their computer but before it was
encrypted at a gateway according to Paul. An encryption product may also not
be practical when mobile users need to send e-mail from outside the corporate
network.* [3]

## [edit] See also

  * Cryptography
  * Cold boot attack
  * Encryption software

  * Cipher
  * Key
  * Famous ciphertexts

<IMG> Cryptography portal  
  * Disk encryption
  * Secure USB drive
  * Secure Network Communications

  
## [edit] References

  1. ^ 2008 CSI Computer Crime and Security Survey, by Robert Richardson, p19
  2. ^ Fiber Optic Networks Vulnerable to Attack, Information Security Magazine, November 15, 2006, Sandra Kay Miller
  3. ^ [1]

  * Helen FouchÃ© Gaines, âCryptanalysisâ, 1939, Dover. ISBN 0-486-20097-3
  * David Kahn, The Codebreakers - The Story of Secret Writing (ISBN 0-684-83130-9) (1967)
  * Abraham Sinkov, Elementary Cryptanalysis: A Mathematical Approach, Mathematical Association of America, 1966. ISBN 0-88385-622-0

## [edit] External links

  * [2]

<IMG>

Look up encryption in Wiktionary, the free dictionary.

  * SecurityDocs Resource for encryption whitepapers
  * Accumulative archive of various cryptography mailing lists. Includes Cryptography list at metzdowd and SecurityFocus Crypto list.

v â¢ d â¢ e

Cryptography  
History of cryptography Â· Cryptanalysis Â· Cryptography portal Â· Topics in
cryptography  
Symmetric-key algorithm Â· Block cipher Â· Stream cipher Â· Public-key
cryptography Â· Cryptographic hash function Â· Message authentication code Â·
Random numbers Â· Steganography  
Retrieved from "http://en.wikipedia.org/wiki/Encryption"

Categories: Cryptography

##### Views

  * Article
  * Discussion
  * Edit this page
  * History

##### Personal tools

  * Log in / create account

##### Navigation

  * Main page
  * Contents
  * Featured content
  * Current events
  * Random article

##### Search



##### Interaction

  * About Wikipedia
  * Community portal
  * Recent changes
  * Contact Wikipedia
  * Donate to Wikipedia
  * Help

##### Toolbox

  * What links here
  * Related changes
  * Upload file
  * Special pages
  * Printable version
  * Permanent link
  * Cite this page

##### Languages

  * Ø§ÙØ¹Ø±Ø¨ÙØ©
  * Bosanski
  * Dansk
  * Deutsch
  * Eesti
  * EspaÃ±ol
  * Esperanto
  * FranÃ§ais
  * Bahasa Indonesia
  * Ãslenska
  * Bahasa Melayu
  * Nederlands
  * æ¥æ¬èª
  * Polski
  * Ð ÑÑÑÐºÐ¸Ð¹
  * Simple English
  * Svenska
  * à¹à¸à¸¢
  * Tiáº¿ng Viá»t
  * Ð£ÐºÑÐ°ÑÐ½ÑÑÐºÐ°
  * ä¸­æ

Powered by MediaWiki

Wikimedia Foundation

  * This page was last modified on 5 April 2009, at 23:58.
  * All text is available under the terms of the GNU Free Documentation License. (See Copyrights for details.)   
Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a U.S.
registered 501(c)(3) tax-deductible nonprofit charity.  

  * Privacy policy
  * About Wikipedia
  * Disclaimers



"""
    cls = Clean_Text()
    doc = doc.split("\n")
    doc,filtered_categories,filtered_hid_categories = cls.filter_html_categories_reverse(doc)
    doc_spacy = clean_wiki(doc)
    logger.debug(doc_spacy)
    logger.debug(filtered_categories)
    logger.debug(filtered_hid_categories)


if __name__ == "__main__":
    main()
