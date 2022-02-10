'''
there's already libraries for parsing addresses

https://github.com/openvenues/libpostal
from postal.parser import parse_address
parse_address('The Book Club 100-106 Leonard St Shoreditch London EC2A 4RH, United Kingdom')


This .py is pretty light on requirements:



https://github.com/nmslib/nmslib/blob/master/python_bindings/README.md
https://towardsdatascience.com/how-to-build-a-smart-search-engine-a86fca0d0795
'''

'''
from a fresh environment:
conda install python
pip install jupyterlab


Installation:
conda install gensim
conda install -c anaconda numpy
conda install -c anaconda nltk
conda install -c anaconda requests
conda install -c conda-forge sentencepiece
conda install -c conda-forge unidecode 
pip install python-Levenshtein
pip install rank_bm25
pip install nmslib




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
        (this whole process usually takes around 60 seconds)
    4. We then merge by IDs using SQLite (which takes around 20 seconds)
    5. Then, if similarity analysis is enabled:
        5.1 We start by creating a system that can efficiently match addresses (i.e., the documents), which takes around 90 seconds:
            This is based on https://towardsdatascience.com/how-to-build-a-search-engine-9f8ffa405eac
            The results here could be improved, the most similar vector is often not really similar to the address
            But the model reduces similarity search from q^r (exponential) to r*1q (linear) which makes this much more efficient.
            Where q=N query entries, r= N reference entries. The query and reference entries can be either file, I prefer to use file1 as a reference since it's better formatted
            But file2 has more entries which could be better for training
        
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
            
            This step took around 200 seconds in my local machine with 24 cores (merging around 3000 entries)
            
        5.3 All similar pairs are merged in the SQLite database
        5.4 All similarity scores are exported to the scores.tsv
    6. The merged addresses are exported from the SQLite database to the <file_output>

Summary of development:
    My initial approach was to use a classical tokenization approach, but the addresses in file2 are quite poorly formatted, so I switched to sentence piece

    I initially tried to do this through the classical data science approach of using pandas, but merging IDs was too slow (couldn't even finish)
    Also, I was basing my implementation on sparse_dot_topn (https://github.com/ing-bank/sparse_dot_topn) for similarity analysis. This method does an all-v-all analysis which was wasting a lot of time

    then I started using sqlite for indexation and merging IDs was much faster
    then I tried the basic tf-idf cosine similarity approach, but it was also quite slow since there's too many entries to compare (exponential scaling)

    then I switched to a more complex approach:
        word embeddings, which basically encodes documents so that tokens that appear in similar contexts are represented by similarly encoded vectors (basically placing a token on a graph with very high dimension, i.e., the <vocab_size>)
        bm25 to extract weighted vectors (somewhat like tf-idf but a somewhat more complex equation)
        nmslib to very efficiently extract be find the closest neighbors (i.e., the reference) of encoded vectors (i.e., the query)

        This approach was adapted from https://towardsdatascience.com/how-to-build-a-smart-search-engine-a86fca0d0795
        Unfortunately the results from this approach were not amazing, a lot of the retrieved neighbors were actually poor matches
        Nonetheless, this can be used to reduce the search space by a lot, now instead of O(c^n) we have O(n)

        So, after this is done we still do a proper similarity analysis between the addresses.

What could be improved?
remove street types from address

'''



import re
from string import punctuation
from Levenshtein import ratio as Levenshtein_ratio
from unidecode import unidecode
import io
import os
import re
import shutil
import sentencepiece as spm
from nltk.cluster.util import cosine_distance
from pickle import load,dump
from pathlib import Path
import sqlite3
from time import time,sleep
import requests
import urllib.request as request
from gzip import open as gzip_open
#similarity analysis takes a while, so we will use multiprocessing
from multiprocessing import Process, current_process, cpu_count, Manager
from string import digits

from gensim.models.fasttext import FastText
from rank_bm25 import BM25Okapi
import numpy as np
import pickle
from nmslib import init as nmslib_init
from nltk.corpus import stopwords

#this would be in an utils.py but we leave it here for the current exercise
postal_code_pattern=re.compile('\d{3,}')
digit_pattern=re.compile('\d+')
city_pattern=re.compile('[A-ZÀ-ú]{2,}(\s[A-ZÀ-ú]{2,})*')
acronyms_pattern=re.compile('\([A-Z]+[\'\"]?\)')
#remove de du des
d_pattern=re.compile('\sd[aeu]?s?\s')
#removing a au aux
a_pattern=re.compile('\sau?x?\s')
#removing l le la les las
l_pattern=re.compile('\sl[ae]?[sz]?\s')
l_pattern_start=re.compile('l[ae]?[sz]?\s')
punctuation_set=punctuation
punctuation_table = str.maketrans('','',punctuation_set)
fr_stopwords=stopwords.words('french')

street_types_dict = {
    "r": "rue",
    "rte": "rue",
    "av": "avenue",
    "crs": "cours",
    "che": "chem",
    "esp": "esplanade",

}

street_types_patterns={
    i:re.compile(f'(\s?|^[A-Za-z]){i}\s', flags=re.IGNORECASE) for i in street_types_dict
}


#helper functions to download and extract training french addresses data
class Downloader():

    def download_file_http(self,url, file_path, c, ctx):
        if c > 5:
            self.download_file_http_failsafe(url, file_path, ctx)
        else:
            if ctx:
                with requests.get(url, stream=True, verify=False) as r:
                    with open(file_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
            else:
                with requests.get(url, stream=True) as r:
                    with open(file_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)

    # slower but safer
    def download_file_http_failsafe(self,url, file_path, ctx):
        with requests.Session() as session:
            if ctx: session.verify = False
            get = session.get(url, stream=True)
            if get.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in get.iter_content(chunk_size=1024):
                        f.write(chunk)

    def download_file_ftp(self,url, file_path, ctx):
        with closing(request.urlopen(url, context=ctx)) as r:
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(r, f)

    def download_file(self,url, output_folder='', stdout_file=None, retry_limit=10):
        file_path = output_folder + url.split('/')[-1]
        ctx = None
        try:
            target_file = request.urlopen(url)
        except:
            try:
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                target_file = request.urlopen(url, context=ctx)
            except:
                print('Cannot download target url', url)
                return
        target_size = target_file.info()['Content-Length']
        transfer_encoding = target_file.info()['Transfer-Encoding']
        if target_size: target_size = int(target_size)
        if os.path.exists(file_path):
            if transfer_encoding == 'chunked':
                return
            elif os.stat(file_path).st_size == target_size:
                print('Not downloading from ' + url + ' since file was already found!', flush=True, file=stdout_file)
                return
            else:
                os.remove(file_path)
        print('Downloading from ' + url + '. The file will be kept in ' + output_folder, flush=True, file=stdout_file)
        c = 0
        while c <= retry_limit:
            if 'ftp' in url:
                try:
                    self.download_file_ftp(url, file_path, ctx)
                except:
                    try:
                        self.download_file_http(url, file_path, c, ctx)
                    except:
                        pass
            else:
                try:
                    self.download_file_http(url, file_path, c, ctx)
                except:
                    pass
            if transfer_encoding == 'chunked': return
            if os.path.exists(file_path):
                if os.stat(file_path).st_size == target_size: return
            c += 1
        print('Did not manage to download the following url correctly:\n' + url)
        raise Exception

    def gunzip(self,source_filepath, dest_filepath=None, block_size=65536):
        if not dest_filepath:
            dest_filepath = source_filepath.strip('.gz')
        if os.path.isdir(dest_filepath):
            file_name = source_filepath.split(SPLITTER)[-1].replace('.gz', '')
            dest_filepath = add_slash(dest_filepath) + file_name
        print('Gunzipping ', source_filepath, 'to', dest_filepath)
        with gzip_open(source_filepath, 'rb') as s_file, \
                open(dest_filepath, 'wb') as d_file:
            while True:
                block = s_file.read(block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)

class Pre_Processor():

    def remove_single_letters(self,input_string):
        res=input_string.split()
        res=[i for i in res if len(i)>1]
        res=' '.join(res)
        return res

    def remove_pattern_start(self,input_string, pattern):
        if not input_string: return ''
        res = str(input_string)
        search = pattern.match(res)
        if search:
            res = res.replace(search.group(), ' ')
        return res

    def remove_pattern(self,input_string, pattern):
        if not input_string: return ''
        res = str(input_string)
        search = pattern.findall(res)
        for s in search:
            res = res.replace(s, ' ')
        return res

    def remove_stop_words(self,input_string):
        res=input_string.split()
        res=[i for i in res if i not in fr_stopwords]
        res=' '.join(res)
        res=res.strip()
        return res


    def preprocess_string(self,input_string):
        if not input_string: return ''
        temp_str = input_string.lower()
        temp_str=temp_str.strip()
        temp_str=temp_str.translate(self.punctuation_table)
        #removing accents
        temp_str=unidecode(temp_str)
        temp_str=self.remove_stop_words(temp_str)
        temp_str=temp_str.split()
        temp_str=' '.join(temp_str)
        if not temp_str: return ''
        return temp_str


    #Processing source 1
    def preprocess_source1(self,yield_line,to_train=False):

        yield_line = [i.strip() for i in yield_line]
        yield_line = [i.strip('"') for i in yield_line]
        entry_id, name, street_number, street_type, street_name, address2, postal_code, city = yield_line
        address = f'{street_type} {street_name}'
        city=self.remove_pattern(city,digit_pattern)
        res = {
            'rawid1': entry_id,
            'name': self.preprocess_string(name),
            'street_number': self.preprocess_string(street_number),
            'address': self.preprocess_string(address),
            'postal_code': self.preprocess_string(postal_code),
            'city': self.preprocess_string(city),
            # too much random input to be reliable but could be used for similarity analysis
            'extra': self.preprocess_string(address2),
        }
        if not to_train:
            if not self.skip_similarity_analysis:
                res['tokenized']= self.tokens_splitter.join(self.spm_str_encoding(address))
        return res

    #Processing source 2

    def process_address2(self,address):

        if not address:
            return '','','',''
        temp_str = str(address)

        # removing acronyms at the end of the address
        self.remove_pattern(temp_str,acronyms_pattern)
        if ',' in temp_str:
            city_postal_code = temp_str.split(',')[-1].strip()
            postal_code = postal_code_pattern.search(city_postal_code)
            if postal_code:
                postal_code = postal_code.group().strip()
                city = city_postal_code.replace(postal_code, '')
            else:
                city = city_postal_code.strip()
            temp_str = temp_str.replace(city_postal_code, '')
            temp_str = temp_str.replace(',', '').strip()
        else:
            # getting city in caps and starting search at the end of the string
            city = city_pattern.match(temp_str[::-1])
            if city:
                city = city.group()[::-1].upper().strip()
                temp_str = temp_str.replace(city, '').strip()
            # getting postal code
            postal_code = postal_code_pattern.match(temp_str[::-1])
            if postal_code:
                postal_code = postal_code.group()
        temp_str = temp_str.strip()
        #we get the first digits in the address
        street_number=digit_pattern.search(temp_str)
        if street_number:
            # but we reject those that are positioned way too far from the begginning of the address e.g.:
            # in "avenue 1954", the "1954" is rejected
            # but in "14 avenue 1954" the "14" is found and kept.
            end_position=street_number.span()[1]
            street_number=street_number.group()
            if end_position>=len(temp_str)/2:
                street_number=''
        else:
            street_number=''
        temp_str=temp_str.replace(street_number,'')
        temp_str = temp_str.strip()
        temp_str = self.standardize_street_type(temp_str)
        temp_str = temp_str.strip()
        return city, postal_code, temp_str,street_number

    #it might be better to use this, but we would need to have a more comprehensive conversion dict.
    #Since i dont know french acronyms that well this would take a while, plus SPM does a pretty good job anyway
    def standardize_street_type(self,address):
        res = str(address)
        street_type = None
        for street_t in street_types_dict:
            regex_pattern = street_types_patterns[street_t]
            pattern_search = regex_pattern.search(address)
            if pattern_search:
                current_str = pattern_search.group()
                street_type = street_types_dict[street_t]
                res = res.replace(current_str, f' {street_type} ')
        return res


    def preprocess_source2(self,yield_line,to_train=False):

        yield_line = [i.strip() for i in yield_line]
        yield_line = [i.strip('"') for i in yield_line]
        #not sure if there's a point to save these
        if len(yield_line) == 1: return
        if len(yield_line) == 4:
            address, website, entry_id, name = yield_line
            address2 = ''
        elif len(yield_line) == 5:
            address, website, entry_id, name, address2 = yield_line
        if not address: address = ''
        city, postal_code, address,street_number = self.process_address2(address)
        city=self.remove_pattern(city,digit_pattern)
        res = {
            'rawid2': entry_id,
            'name': self.preprocess_string(name),
            'street_number': self.preprocess_string(street_number),
            'address': self.preprocess_string(address),
            'postal_code': self.preprocess_string(postal_code),
            'city': self.preprocess_string(city),
            'extra': self.preprocess_string(address2),
        }
        if not to_train:
            if not self.skip_similarity_analysis:
                res['tokenized']= self.tokens_splitter.join(self.spm_str_encoding(address))
        return res

    def parse_tsv(self,tsv_path, line_processing_function,to_train=False):
        with open(tsv_path) as file:
            # skipping headers
            file.readline()
            for line in file:
                line = line.strip('\n')
                line = line.strip('"')
                if line:
                    line = line.split('\t')
                    current_info = line_processing_function(line,to_train)
                    if current_info:
                        yield current_info

class SPM_Tokenizer(Downloader):
    '''
    this class is a tokenizer based on data from https://adresse.data.gouv.fr/
    based on: https://github.com/google/sentencepiece

    this is a neat way to tokenize data without spending too much time on pre-processing
    it's also language independent which is great for dealing with french addresses

    '''
    def __init__(self,file1,file2,out_folder=None,vocab_size=20000):

        if not out_folder:
            self.out_folder=os.getcwd()+'/'
        else: self.out_folder=out_folder+'/'
        self.punctuation_table = str.maketrans(punctuation_set, ' ' * len(punctuation))
        if not self.skip_similarity_analysis:
            start=time()
            print('Downloading data and generating sentencepiece model')
            self.model_folder = f'{self.out_folder}address_sim_model/'
            if not os.path.exists(self.model_folder):
                Path(self.model_folder).mkdir(parents=True, exist_ok=True)
            self.download_fr_addresses(self.model_folder)
            #we train SPM with all data
            training_data = {'file1': file1, 'file2': file2}
            training_data['fr_addresses']= f'{self.model_folder}fr_addresses.csv'
            self.training_data=training_data
            #tokenization model
            self.spm_model_name = f'{self.model_folder}spm'
            self.spm_model_path=f'{self.spm_model_name}.model'
            self.sentence_piece_input_path = f'{self.model_folder}spm_input'
            self.vocab_size=vocab_size
            self.spm_model=self.generate_sentencepiece_model()
            print(f'Finished download data and generating sentencepiece model in {time()-start} seconds')




    def download_fr_addresses(self,out_folder):
        url='https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/adresses-france.csv.gz'
        infile=f'{out_folder}adresses-france.csv.gz'
        outfile=f'{out_folder}fr_addresses.csv'
        if not os.path.exists(outfile):
            self.download_file(url,out_folder)
            if not os.path.exists(outfile):
                self.gunzip(infile,outfile)


    def generate_sentencepiece_input(self):
        print('Generating Sentencepiece input file')
        generator1 = self.parse_tsv(self.training_data['file1'], self.preprocess_source1,to_train=True)
        generator2 = self.parse_tsv(self.training_data['file2'], self.preprocess_source2,to_train=True)
        with open(self.sentence_piece_input_path,'w+') as outfile:
            for street_number,street_name,postal_code,city,old_city in self.yield_fr_addresses():
                address=self.preprocess_string(street_name)
                outfile.write(f'{address}\n')
            for generator in [generator1,generator2]:
                for entry_dict in generator:
                    street_name=entry_dict['address']
                    street_name=self.preprocess_string(street_name)
                    outfile.write(f'{street_name}\n')

    def generate_sentencepiece_model(self):
        if not os.path.exists(self.sentence_piece_input_path):
            self.generate_sentencepiece_input()
        model_type='unigram'
        if not os.path.exists(self.spm_model_path):
            spm.SentencePieceTrainer.train(input=self.sentence_piece_input_path,
                                           model_prefix=self.spm_model_name,
                                           vocab_size=self.vocab_size,
                                           model_type=model_type,
                                           character_coverage=0.9995,
                                           )
            print('Sentencepiece model created, saving...')
            model = spm.SentencePieceProcessor(model_file=self.spm_model_path)
            return model
        else:
            print('Sentencepiece model already present, loading...')
            model = spm.SentencePieceProcessor(model_file=self.spm_model_path)
            return model

    #trainding data containing a lot of french addresses - 25644170
    def yield_fr_addresses(self):
        with open(self.training_data['fr_addresses']) as file:
            file.readline()
            for line in file:
                line=line.strip('\n')
                line=line.split(';')
                try:
                    street_number=line[2]
                    street_name=line[4]
                    postal_code=line[5]
                    city=line[7]
                    old_city=line[9]
                    yield street_number,street_name,postal_code,city,old_city
                except:
                    pass

    def spm_str_encoding(self,string):
        return self.spm_model.encode(self.preprocess_string(string), out_type=str)

class Similarity_Model():
    # this is based on https://towardsdatascience.com/how-to-build-a-smart-search-engine-a86fca0d0795

    def generate_word_embedding_model(self):
        model_path=f'{self.model_folder}word_embedding.model'
        if not os.path.exists(model_path):
            print('Getting training tokenized vectors')
            vectors_list = self.get_training_vectors()
            print('Generating word embedding model (gensim.FastText)')
            #this requires quite a bit of RAM
            res = FastText(
                sg=1,  # use skip-gram: usually gives better results
                window=5,  # window size: 10 tokens before and 10 tokens after to get wider context
                min_count=5,  # only consider tokens with at least n occurrences in the corpus
                negative=15,  # negative subsampling: bigger than default to sample negative examples more
                min_n=3,  # min character n-gram
                max_n=6,  # max character n-gram
                workers=self.worker_count  # max character n-gram
            )
            res.build_vocab(vectors_list)
            res.train(vectors_list,epochs=6,total_examples=res.corpus_count,total_words=res.corpus_total_words)
            res.save(model_path)

    def load_word_embedding_model(self,):
        model_path=f'{self.model_folder}word_embedding.model'
        return FastText.load(model_path)  # load

    def generate_weighted_vectors(self,word_embedding_model):
        print('Generating weighted vectors (bm25)')
        #now what we can embed vectors, we create a dataset to actually check against
        #we use table 1 since it's better formated
        vectors_generator=self.get_unmatched_tokenized_vectors(self.reference_table)
        #needs to be a list for bm25
        vectors_list=[]
        self.vectors_to_sql={}
        c=0
        for sql_id,token_list in vectors_generator:
            self.vectors_to_sql[c]=sql_id
            c+=1
            vectors_list.append(token_list)
        # https://pypi.org/project/rank-bm25/
        bm25 = BM25Okapi(vectors_list)
        weighted_doc_vects = []
        for i, doc in enumerate(vectors_list):
            doc_vector = []
            for word in doc:
                vector = word_embedding_model.wv[word]
                weight = (bm25.idf[word] * ((bm25.k1 + 1.0) * bm25.doc_freqs[i][word]))/(bm25.k1 * (1.0 - bm25.b + bm25.b * (bm25.doc_len[i] / bm25.avgdl)) + bm25.doc_freqs[i][word])
                weighted_vector = vector * weight
                doc_vector.append(weighted_vector)
            doc_vector_mean = np.mean(doc_vector, axis=0)
            weighted_doc_vects.append(doc_vector_mean)
        return weighted_doc_vects


    def generate_similarity_search_model(self,weighted_doc_vects):
        print('Generating similarity search model (nmslib)')
        # https://github.com/nmslib/nmslib
        # create a matrix from our document vectors
        data = np.vstack(weighted_doc_vects)
        # initialize a new index, using a HNSW index on Cosine Similarity
        #this is generally what is used, but one could try other methods, e.g., manhattan.
        #https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
        nmslib_model = nmslib_init(method='hnsw', space='cosinesimil')
        nmslib_model.addDataPointBatch(data)
        nmslib_model.createIndex({'post': 2}, print_progress=False)
        return nmslib_model

    def get_most_similar(self,query_vectors):
        query_list=[]
        query_ids=[]
        for table_id,tokens_list in query_vectors:
            #getting embedding vector per token
            query = [self.word_embedding_model.wv[vec] for vec in tokens_list]
            query = np.mean(query, axis=0)
            query_list.append(query)
            query_ids.append(table_id)
        #getting most similar neighbor
        #we could have k>1 and have multiple matches per entry, but this increase time by k times
        k_neighbours = self.similarity_search_model.knnQueryBatch(query_list, k=1,num_threads=self.worker_count)
        c=0
        similarity_threshold=self.similarity_threshold['model']
        for neighbour in k_neighbours:
            query_table_id=query_ids[c]
            matches_ids,matches_scores=neighbour
            match_id=matches_ids[0]
            reference_table_id=self.vectors_to_sql[match_id]
            #just to standardize it, 0 min , 1 max
            match_score=1-matches_scores[0]
            if match_score>similarity_threshold:
                yield query_table_id,reference_table_id,match_score
            c+=1

    def get_training_vectors(self):
        tokenized_vectors=[]
        for tokenized in self.get_tokenized_vectors('table1'):
            tokenized_vectors.append(tokenized)
        for street_number, street_name, postal_code, city, old_city in self.yield_fr_addresses():
            address = f'{street_number} {street_name}'
            t_vector = self.spm_str_encoding(address)
            tokenized_vectors.append(t_vector)
        return tokenized_vectors

    def generate_embedding_and_similarity_models(self):
        #the embedding is working pretty well
        self.generate_word_embedding_model()
        word_embedding_model = self.load_word_embedding_model()
        weighted_doc_vects = self.generate_weighted_vectors(word_embedding_model)
        similarity_search_model = self.generate_similarity_search_model(weighted_doc_vects)
        self.word_embedding_model=word_embedding_model
        self.similarity_search_model=similarity_search_model

class Similarity_Analysis():

    def processes_handler(self,target_worker_function, add_sentinels=True):
        '''
        this will first generate one process per worker, then we add sentinels to the end of the list which will basically tell us when the queue is empty
        if we need to add new work (e.g. when doing taxa annotation) we just add the new work to the start of the list
        '''
        # os.getpid to add the master_pid
        if len(self.queue)<self.worker_count: worker_count=len(self.queue)
        else: worker_count=self.worker_count
        processes = [Process(target=target_worker_function, args=(self.queue, os.getpid(),)) for _ in range(worker_count)]
        # adding sentinel record since queue can be signaled as empty when its really not
        if add_sentinels:
            for _ in range(worker_count):   self.queue.append(None)
        for process in processes:
            process.start()
        # we could manage the processes memory here with a while cycle
        for process in processes:
            process.join()
            # exitcode 0 for sucessful exists
            if process.exitcode != 0:
                sleep(5)
                print('Ran into an issue, check the log for details. Exitting!')
                os._exit(1)

    def similarity_analysis_worker_function(self,queue, master_pid):
        while True:
            record = queue.pop(0)
            if record is None: break
            query_entry,reference_entry,match_score=record
            self.run_similarity_analysis(query_entry,reference_entry,match_score)

    def yield_to_merge_by_similarity(self):
        print('Generating list of potentially similar addresses')
        query_vectors = self.get_unmatched_tokenized_vectors(self.query_table)
        #contains entries with query_table_id,reference_table_id
        similarity_generator=self.get_most_similar(query_vectors)
        #this will contain the ids of each table, it's used to retrieve data from sqlite
        reference_entries=[]
        query_entries=[]
        #this will contain the ids pairs to test
        ids_to_test=[]
        #getting all entries sql IDs
        for query_table_id,reference_table_id,match_score in similarity_generator:
            reference_entries.append(reference_table_id)
            query_entries.append(query_table_id)
            ids_to_test.append([query_table_id,reference_table_id,match_score])
        query_entries_dict={}
        reference_entries_dict={}
        #getting all entries data (e.g., address and city)
        for table_id,entry_dict in self.fetch_entries(self.reference_table,reference_entries):
            reference_entries_dict[table_id]=entry_dict
        for table_id,entry_dict in self.fetch_entries(self.query_table,query_entries):
            query_entries_dict[table_id]=entry_dict

        ids_to_test_entries=[]
        for query_table_id,reference_table_id,match_score in ids_to_test:
            reference_entry=reference_entries_dict[reference_table_id]
            query_entry=query_entries_dict[query_table_id]
            ids_to_test_entries.append([query_entry,reference_entry,match_score])

        print(f'Performing similarity analysis on {len(ids_to_test_entries)} entries')
        self.queue.extend(ids_to_test_entries)
        #now each worker will perform similarity analysis from the pool of pairs
        self.processes_handler(self.similarity_analysis_worker_function)
        self.scores=[]
        while self.mp_results:
            merged_entry,scores_entry = self.mp_results.pop(0)
            #would remove this line after proper benchmarking, also after optimizing similarity analysis
            self.scores.append(scores_entry)
            yield merged_entry

    def fuzzy_matching(self,string1, string2):
        #number of changes but taking not taking into account repetitions or order of tokens
        if not string1 or not string2: return 0
        if isinstance(string1,list) or isinstance(string1,set): set1=set(string1)
        else:                        set1 = set(string1.split())
        if isinstance(string2,list) or isinstance(string2,set): set2=set(string2)
        else:                        set2 = set(string2.split())
        if not set1.intersection(set2): return 0
        ordered_list1 = sorted(set1)
        ordered_list2 = sorted(set2)
        ordered_string1 = ' '.join(ordered_list1)
        ordered_string2 = ' '.join(ordered_list2)
        #also tried with edlib but it was actually around ten times slower
        return Levenshtein_ratio(ordered_string1, ordered_string2)

    def jaccard_distance(self,label1, label2):
        #similarity between two sets
        if not label1 or not label2: return 1
        if not label1 and not label2: return 1
        if isinstance(label1,str): label1={label1}
        if isinstance(label2,str): label2={label2}
        union_labels=len(label1.union(label2))
        intersection_labels=len(label1.intersection(label2))
        if not union_labels: return 0
        return 1-(union_labels - intersection_labels) / union_labels

    def is_equal(self,string1,string2):
        if not string1 or not string2:
            res = 1
        elif string1 == string2:
            res = 1
        else:
            res = 0
        return res

    def run_similarity_analysis(self,query_entry,reference_entry,match_score):
        # this could be improved by using more efficient implementations
        #C implementations of jaccard_distance, Levenshtein_ratio, and ordered set Levenshtein_ratio?

        table_number_query=self.query_table[-1]
        rawid_query = query_entry[f'rawid{table_number_query}']
        city_query = query_entry['city']
        address_query = query_entry['address']
        #address_query=self.remove_single_letters(address_query)
        street_number_query = query_entry['street_number']
        postal_code_query = query_entry['postal_code']
        name_query = query_entry['name']
        extra_query = query_entry['extra']
        tokenized_query = query_entry['tokenized']

        table_number_reference=self.reference_table[-1]
        rawid_reference = reference_entry[f'rawid{table_number_reference}']
        city_reference = reference_entry['city']
        street_number_reference = reference_entry['street_number']
        address_reference = reference_entry['address']
        #address_reference=self.remove_single_letters(address_reference)
        postal_code_reference = reference_entry['postal_code']
        name_reference = reference_entry['name']
        extra_reference = reference_entry['extra']
        tokenized_reference = reference_entry['tokenized']

        # city needs to be similar
        city_score = self.fuzzy_matching(city_query, city_reference)

        if city_score > self.similarity_threshold['city']:
            #the model is not very reliable so we also use fuzzy matching
            address_score = self.fuzzy_matching(tokenized_query, tokenized_reference)
            postal_code_score=self.is_equal(postal_code_query,postal_code_reference)
            name_score = self.fuzzy_matching(name_query,name_reference)
            street_number_score =self.is_equal(street_number_query,street_number_query)



            if  address_score       >=  self.similarity_threshold['address']        and \
                postal_code_score   >=  self.similarity_threshold['postal_code']    and \
                name_score          >=  self.similarity_threshold['name']           and \
                street_number_score >=  self.similarity_threshold['street_number']  :
                merged_entry = {
                    f'table{table_number_query}_rawid{table_number_query}': rawid_query,
                    f'table{table_number_reference}_rawid{table_number_reference}': rawid_reference,
                    f'table{table_number_query}_name': name_query,
                    f'table{table_number_reference}_name': name_reference,
                    f'table{table_number_query}_address': address_query,
                    f'table{table_number_reference}_address': address_reference,
                    f'table{table_number_query}_street_number': street_number_query,
                    f'table{table_number_reference}_street_number': street_number_reference,
                    f'table{table_number_query}_postal_code': postal_code_query,
                    f'table{table_number_reference}_postal_code': postal_code_reference,
                    f'table{table_number_query}_city': city_query,
                    f'table{table_number_reference}_city': city_reference,
                    f'table{table_number_query}_extra': extra_query,
                    f'table{table_number_reference}_extra': extra_reference,
                    f'table{table_number_query}_tokenized': self.tokens_splitter.join(tokenized_query),
                    f'table{table_number_reference}_tokenized': self.tokens_splitter.join(tokenized_reference),
                }
                scores_entry = {'city': [city_query, city_reference, city_score],
                                'model': [tokenized_query, tokenized_reference, match_score],
                                'address': [address_query, address_reference, address_score],
                                'postal_code': [postal_code_query, postal_code_reference, postal_code_score],
                                'name': [name_query, name_reference, name_score],
                                'street_number': [street_number_query, street_number_reference, street_number_score],
                                }
                #here we add it to a list that is shared between all workers
                self.mp_results.append([merged_entry,scores_entry])

class SQLITE_Connector():
    def __init__(self):
        self.query_step=10000
        self.tokens_splitter='##'
        self.db = f'address.db'
        self.tables={
            #we assume we always have the id, even though some entries dont
            'table1':['rawid1','name','street_number','address','postal_code','city','extra'],
            'table2':['rawid2','name','street_number','address','postal_code','city','extra'],
        }
        if not self.skip_similarity_analysis:
            self.tables['table1'].append('tokenized')
            self.tables['table2'].append('tokenized')

        merged_headers=[]
        for table in self.tables:
            for header in self.tables[table]:
                current_header=f'{table}_{header}'
                merged_headers.append(current_header)
        self.tables['merged']=merged_headers



    def start_sqlite_cursor(self):
        self.sqlite_connection = sqlite3.connect(self.db)
        self.cursor = self.sqlite_connection.cursor()

    def commit(self):
        self.sqlite_connection.commit()

    def execute(self, command):
        return self.cursor.execute(command)

    def executemany(self, command, chunk):
        return self.cursor.executemany(command, chunk)

    def commit_and_close_sqlite_cursor(self):
        self.sqlite_connection.commit()
        self.sqlite_connection.close()

    def close_sql_connection(self):
        self.sqlite_connection.close()

    def generate_inserts(self, data_yielder):
        step=self.query_step
        temp=[]
        for i in data_yielder:
            if len(temp)<step:
                temp.append(i)
            else:
                yield temp
                temp=[]
                temp.append(i)
        yield temp

    def create_main_tables(self):
        for current_table in self.tables:
            main_id = 'id'
            #creating table
            db_headers=self.tables[current_table]
            create_table_command = f'CREATE TABLE {current_table} ({current_table}_{main_id} integer primary key, '
            for header in db_headers:
                if header=='postal_code':
                    create_table_command += f'{header} INT, '
                else:
                    create_table_command += f'{header} TEXT, '
            create_table_command = create_table_command.rstrip(', ')
            create_table_command += ')'
            self.execute(create_table_command)
            self.commit()
            #indexing table
            create_index_command = f'CREATE INDEX {current_table}_{main_id}x ON {current_table} ({current_table}_{main_id})'
            self.execute(create_index_command)
            self.commit()

    def generate_insert_command(self,table_name):
        db_headers = self.tables[table_name]
        headers_str=', '.join(db_headers)
        headers_str=f'({headers_str})'
        n_values=['?' for i in range(len(db_headers))]
        n_values_str=', '.join(n_values)
        n_values_str=f'({n_values_str})'
        insert_command = f'INSERT INTO {table_name} {headers_str} values {n_values_str}'
        return insert_command

    def convert_dict_to_sql(self, row_dict, target_table):
        res=[]
        for db in self.tables[target_table]:
            res.append(row_dict[db])
        return res

    def yield_data(self,generator,current_table):
        db_headers = self.tables[current_table]
        for entry in generator:
            yield self.convert_dict_to_sql(entry,current_table)

    def create_sql_db(self):
        if os.path.exists(self.db):
            os.remove(self.db)
        self.start_sqlite_cursor()
        self.create_main_tables()
        generator1 = self.parse_tsv(file1, self.preprocess_source1)
        generator2 = self.parse_tsv(file2, self.preprocess_source2)
        generators_dict={'table1': generator1, 'table2': generator2}
        for table_name in generators_dict:
            insert_command = self.generate_insert_command(table_name)
            generator=generators_dict[table_name]
            for entry in generator:
                data_yielder = self.yield_data(generator, table_name)
                if data_yielder:
                    generator_insert = self.generate_inserts(data_yielder)
                    for table_chunk in generator_insert:
                        self.executemany(insert_command, table_chunk)
                    self.commit()

    def yield_fetch(self,to_fetch):
        step=self.query_step
        temp=[]
        for i in to_fetch:
            if len(temp)<step:
                temp.append(i)
            else:
                yield temp
                temp=[]
                temp.append(i)
        yield temp

    def get_tokenized_vectors(self,table):
        fetch_command=f'SELECT {table}_id,tokenized FROM {table}'
        res_fetch = self.execute(fetch_command).fetchall()
        for fetched_entry in res_fetch:
            table_id,tokenized=fetched_entry
            tokenized=tokenized.split(self.tokens_splitter)
            yield tokenized


    def fetch_all_entries(self,table):
        db_headers=self.tables[table]
        db_headers_str=', '.join(db_headers)
        fetch_command=f'SELECT {table}_id,{db_headers_str} FROM {table}'
        res_fetch = self.execute(fetch_command).fetchall()
        for fetched_entry in res_fetch:
            entry_dict={f'{table}_id':fetched_entry[0]}
            for j in range(len(db_headers)):
                current_header=db_headers[j]
                header_entry=fetched_entry[j+1]
                if current_header=='tokenized':
                    header_entry=header_entry.split(self.tokens_splitter)
                entry_dict[current_header]=header_entry
            yield entry_dict

    def fetch_entries(self,table,table_ids):
        db_headers=self.tables[table]
        db_headers_str=', '.join(db_headers)
        for chunk in self.yield_fetch(table_ids):
            chunk_ids=[str(i) for i in chunk]
            chunk_ids=', '.join(chunk_ids)
            chunk_ids=f'({chunk_ids})'
            fetch_command=f'SELECT {table}_id,{db_headers_str} FROM {table} WHERE {table}_id IN {chunk_ids}'
            res_fetch = self.execute(fetch_command).fetchall()
            for fetched_entry in res_fetch:
                table_id=fetched_entry[0]
                entry_dict={f'{table}_id':table_id}
                for j in range(len(db_headers)):
                    current_header=db_headers[j]
                    header_entry=fetched_entry[j+1]
                    if current_header=='tokenized':
                        header_entry=header_entry.split(self.tokens_splitter)
                    entry_dict[current_header]=header_entry
                yield table_id,entry_dict

class Address_Matcher_FR(Similarity_Model,SPM_Tokenizer,Similarity_Analysis,Pre_Processor,SQLITE_Connector):
    def __init__(self,file1,file2,vocab_size=10000,test=False,skip_similarity_analysis=False):
        self.skip_similarity_analysis=skip_similarity_analysis
        self.worker_count=cpu_count()-1
        print(f'Using {self.worker_count} cores for parallelization')
        SQLITE_Connector.__init__(self)
        SPM_Tokenizer.__init__(self,file1,file2,vocab_size=vocab_size)
        self.similarity_threshold={'model':0.9,'address':0.7,'city':0.9,'postal_code':1,'name':0.8,'street_number':1}
        self.test=test
        if self.test:
            self.similarity_threshold={'model':-1,'address':-1,'city':-1,'postal_code':-1,'name':-1,'street_number':-1}
        self.manager = Manager()
        self.queue = self.manager.list()
        self.mp_results = self.manager.list()
        #not sure which one to make query/matching
        #on one hand the table2 has more data, one the other, table1 has cleaner data
        self.query_table='table2'
        self.reference_table='table1'


    def yield_to_merge_by_ID(self):
        self.merged_by_ids=0
        db_headers=[]
        for table in ['table1','table2']:
            for header in self.tables[table]:
                header_str=f'{table}.{header}'
                db_headers.append(header_str)
        db_headers_str=', '.join(db_headers)
        fetch_command=f'SELECT {db_headers_str} FROM table1 INNER JOIN table2 ON table1.rawid1 = table2.rawid2'
        res_fetch = self.execute(fetch_command).fetchall()
        for fetched_entry in res_fetch:
            merged_entry={}
            for j in range(len(db_headers)):
                current_header=db_headers[j].replace('.','_')
                merged_entry[current_header]=fetched_entry[j]
            self.merged_by_ids+=1
            yield merged_entry

    def insert_to_merge(self,yield_function):
        #tried with pandas first but it's extremely slow
        table_name='merged'
        insert_command = self.generate_insert_command(table_name)
        generator=yield_function()
        data_yielder = self.yield_data(generator, table_name)
        if data_yielder:
            generator_insert = self.generate_inserts(data_yielder)
            for table_chunk in generator_insert:
                self.executemany(insert_command, table_chunk)
            self.commit()

    def get_unmatched(self):
        remaining_per_table= {'table1':set(),'table2':set()}
        for table in ['table1','table2']:
            table_number=table[-1]
            fetch_command=f'SELECT {table}_id,rawid{table_number} FROM {table} WHERE rawid{table_number} NOT IN (SELECT {table}_rawid{table_number} FROM merged)'
            res_fetch = self.execute(fetch_command).fetchall()
            for fetched_entry in res_fetch:
                table_id,rawid=fetched_entry
                remaining_per_table[table].add(table_id)
        return remaining_per_table['table1'],remaining_per_table['table2']

    def get_unmatched_tokenized_vectors(self,table):
        table_number=table[-1]
        fetch_command=f'SELECT {table}_id,rawid{table_number},tokenized FROM {table} WHERE rawid{table_number} NOT IN (SELECT {table}_rawid{table_number} FROM merged)'
        res_fetch = self.execute(fetch_command).fetchall()
        for fetched_entry in res_fetch:
            table_id,rawid,tokenized=fetched_entry
            if tokenized:
                tokenized = tokenized.split(self.tokens_splitter)
                yield table_id,tokenized



    def export_scores(self):
        with open('scores.tsv','w+') as file:
            first_line=[]
            for score_entry in self.scores:
                if not first_line:
                    for k in score_entry:
                        first_line.extend([f'{k}_table_1',f'{k}_table_2',f'{k}_score'])
                    first_line='\t'.join(first_line)
                    first_line=f'{first_line}\n'
                    file.write(first_line)
                line=[]
                for k in score_entry:
                    current_data=score_entry[k]
                    line.extend(current_data)
                line=[str(i) for i in line]
                line = '\t'.join(line)
                line = f'{line}\n'
                file.write(line)

    def export_merged_tsv(self,output_file):
        all_entries=self.fetch_all_entries('merged')
        with open(output_file,'w+') as file:
            first_line=[]
            for entry in all_entries:
                if not self.skip_similarity_analysis:
                    entry.pop('table1_tokenized')
                    entry.pop('table2_tokenized')
                if not first_line:
                    first_line.extend(entry.keys())
                    first_line='\t'.join(first_line)
                    first_line=f'{first_line}\n'
                    file.write(first_line)
                line=[str(entry[i]) for i in entry]
                line = '\t'.join(line)
                line = f'{line}\n'
                file.write(line)

    def create_dataframes(self,file_output):

        print('Creating SQL DBs')
        start=time()
        self.create_sql_db()
        print(f'Finished created SQL DBs in {time()-start} seconds')

        #matching by IDs and inserting into merged table
        print('Merging by IDs')
        start=time()
        self.insert_to_merge(self.yield_to_merge_by_ID)
        print(f'Finished merging {self.merged_by_ids} entries by IDs in {time()-start} seconds')


        if not self.skip_similarity_analysis:

            #generating idf for calculasing tf-idf and cosine similarity
            print('Generating embedding and similarity analysis models')
            start=time()
            # for the whole files, this usually takes around 250 seconds
            self.generate_embedding_and_similarity_models()
            print(f'Finished generating embedding and similarity analysis models in {time()-start} seconds')


            #matching remaining entries by similarity and inserting into merged table
            print(f'Performing similarity analysis with {self.worker_count} workers')
            start=time()
            self.insert_to_merge(self.yield_to_merge_by_similarity)

            print(f'Finished merging {len(self.scores)} entries by similarity in {time()-start} seconds')

            start=time()
            #for benchmarking
            print(f'Exporting scores')
            self.export_scores()
        else:
            print('Skipped similarity analysis')
        #exporting merged tsv
        print(f'Exporting merged table')
        self.export_merged_tsv(file_output)
        print(f'Finished exporting data in {time()-start} seconds')




if __name__ == '__main__':
    file_output='merged.tsv'
    #all input files
    fr_addresses = 'fr_addresses.csv'
    file1 = 'source1.tsv'
    file2 = 'source2.tsv'
    #file1 = 'smalltsource1.tsv'
    #file2 = 'smalltsource2.tsv'
    vocab_size=10000
    #test run just reduces all thresholds to -1 so you can check how the similarity analysis is performing
    test_run=False
    # True to merge tables only with IDs (2- 3 minutes), False to perform similarity analysis
    skip_similarity_analysis=False

    start_time=time()
    matcher=Address_Matcher_FR(file1,file2,vocab_size=vocab_size,test=test_run,skip_similarity_analysis=skip_similarity_analysis)
    matcher.create_dataframes(file_output)
    print(f'Total time {time()-start_time} seconds')

