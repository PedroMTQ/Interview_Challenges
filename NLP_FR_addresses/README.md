# NLP_FR_addresses

This was a challenge for a machine learning position focused on NLP.

The idea here is to match two lists of addresses that have different formats and different information.




## Summary

This does not use pandas as it is too slow for such a large scale analysis.
The input files contain in total 5 million entries so I had to adopt highly efficient methods.
In particular the word embedding is used a search space reduction method to avoid doing an all vs all similarity analysis. Instead, all of the documents in file 1 are embedded (FastText model since it deals well with typos) and weighted vectors are created (using bm25 -  an "extension" of TF-IDF). Then each document in file 2 is embedded and we fetch the nearest neighbor using nmslib (keep in mind that just because a document is the nearest neighbor it doesnt mean it's actually the same address).
With this, the search space is decrease from O(c^n) to O(n).
Lastly a similarity search is performed between the nearest neighbor and the document.

This takes around 30 minutes using 23 cores

## Installation
conda install gensim
conda install -c anaconda numpy
conda install -c anaconda nltk
conda install -c anaconda requests
conda install -c conda-forge sentencepiece
conda install -c conda-forge unidecode
pip install python-Levenshtein
pip install rank_bm25
pip install nmslib

## Step by step

how does this work?
    1.custom pre-processing of data to standardize into the same types of entries:
        - rawid (the id in the table)
        - name
        - address
        - postal_code
        - city
        - extra
    2. Train a sentence piece model to tokenize the addresses (makes tokenization less error prone).
        this is trained on fr_addresses.csv and the input files
    3. Creation of SQLite databases with all the previously mentioned types of entries plus "tokenized" version of address
        (this is much faster than using pandas)
    4. We then merge by IDs using SQLite
    5. Then, if similarity analysis is enabled:
        5.1 We start by creating a system that can efficiently match addresses (i.e., the documents), which takes around 90 seconds:
            This is based on https://towardsdatascience.com/how-to-build-a-search-engine-9f8ffa405eac
            The results here could be improved, the most similar vector is often not really similar to the address
            But the model reduces similarity search from q^r (exponential) to r*1q (linear) which makes this much more efficient.
            Where q=N query entries, r= N reference entries. The query and reference entries can be either file.
            5.1.1 we generate a FastText word embedding model (so that similar words have a similar encoding).
                FastText is quite useful for cases where there are spelling errors (due to the n-grams)
                this is trained this on fr_addresses.csv
            5.1.2 we generate the weighted vectors for the reference table using BM25
                i.e., the reference to match against, which can be either the unmatched entries (IDs) from input 1 or 2
            5.1.3 we generate a highly efficient similarity search model using nmslib
            5.1.4 we yield all matching documents above the <model> threshold
        5.2 After retrieving all potential matches, we actually do a more rigorous similarity match taking into account:
            - city: fuzzy similarity (barrier)
            - address: fuzzy similarity
            - postal code: Levenshtein_ratio
            - name: fuzzy similarity
            - street number: binary

        5.3 All similar pairs are merged in the SQLite database
        5.4 All similarity scores are exported to the scores.tsv
    6. The merged addresses are exported from the SQLite database to the <file_output>
