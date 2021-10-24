import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import sys
import os
import string 
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}

    for file in os.scandir(directory):
        with open(file, encoding="utf8") as f:
            # print(f.path, f.name)
            file_name = f.name
            content = f.read()
            files[file_name] = content

    return files



def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopwordslist = stopwords.words('english')
    tokens = word_tokenize(document)
    words = []
    for token in tokens:
        # remove punctuation
        token.translate(str.maketrans('','',string.punctuation))
        if token.isalnum():
            # convert to lowercase
            word = token.lower()
            # remove stopwords
            if word not in stopwordslist:
                words.append(word)
        
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    N = len(documents)
    terms = set()
    _ = [terms.update(words) for key, words in documents.items()]
    # initialize dictionary of all terms with idf 1
    idfs = { term: 1 for term in terms }
    
    for term in terms:
        doc_freq = 1
        for document, words in documents.items():
            if term in words:
                doc_freq += 1
        idf = math.log( N / doc_freq )
        # update term idf value
        idfs[term] = idf
    
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    tf_idfs = { name: 0 for name in files }
    # loop through each file (document)
    for file, words in files.items():
        # initialize frequency of occurence of each word in query to 0
        tfs = { term: 0 for term in query }
        # calculate frequency of occurence of each word in query
        for word in words:
            if word in tfs:
                tfs[word] += 1
        # if word occurs in document, calculate tf_idf by multiplying tf to idf        
        for term, frequency in tfs.items():
            if frequency > 0:
                tf_idf = frequency * idfs[term]
                # add tf_idf value to document's tf_idf total for ranking
                tf_idfs[file] += tf_idf
    # sort document tf_idf by rank (high to low)
    top_files = { name: tf_idfs for name, tfidfs in sorted(
            tf_idfs.items(), 
            key=lambda item: item[1],
            reverse=True 
        ) 
    }
    # top file names in list
    top_files = [file_name for file_name in top_files]
    # return top n files only
    return top_files[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    tf_idfs = { name: 0 for name in sentences }
    # loop through each sentence ()
    for sentence, words in sentences.items():
        # initialize frequency of occurence of each word in query to 0
        tfs = { term: 0 for term in query }
        # calculate frequency of occurence of each word in query
        for word in words:
            if word in tfs:
                tfs[word] += 1
        # if word occurs in sentence, calculate tf_idf by multiplying tf to idf        
        for term, frequency in tfs.items():
            if frequency > 0:
                tf_idf = frequency * idfs[term]
                # add tf_idf value to document's tf_idf total for ranking
                tf_idfs[sentence] += tf_idf
    # sort sentence tf_idf by rank (high to low)
    top_sentences = { name: tf_idfs for name, tfidfs in sorted(
            tf_idfs.items(), 
            key=lambda item: item[1],
            reverse=True 
        ) 
    }
    # top file sentences in list
    top_sentences = [sentence for sentence in top_sentences]
    # return top n sentences only
    return top_sentences[:n]


if __name__ == "__main__":
    main()
