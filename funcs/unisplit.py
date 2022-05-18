import urllib.parse
import urllib.request
from collections import defaultdict
import random 
import itertools

def map_to_uniProtID(query=''):
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
                'from': 'PDB_ID',
                'to': 'ACC',
                'format': 'tab',
                'query': query
    }
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
    result = response.decode('utf-8').split()[2:]
    return list(zip(result,result[1:]))[::2]


def cluster_protein_structures(list_of_protein_structures):
    
    #create query
    query = ' '.join([p for p in list_of_protein_structures])
    
    # map to uniprotIDs
    maps = map_to_uniProtID(query)
    
    # Remove duplicates.
    check_val = set() 
    res = []
    for i in maps:
        if i[0] not in check_val:
            res.append(i)
            check_val.add(i[0]) 
    
    # Assign default key to protein structures with no UniProt ID.
    prot = [x[0] for x in res]
    withoutUniProtID = set(list_of_protein_structures) - set(prot)
    for x in withoutUniProtID:
        res.append((x, "Unknown"))
    
    ## Clusters of UniProt IDs
    clusters = defaultdict(list)
    for v, k in res:
        clusters[k].append(v)
        
    return clusters


def uni_split_data(clusters, split = 0.8):
    #shuffle
    keys = list(clusters.keys())
    random.shuffle(keys)
    cl = dict()
    for key in keys:
        cl.update({key: clusters[key]})
    
    #split dict based on UniProt IDs
    split_point = int (split * len (cl))
    train_dict = dict(list(cl.items())[:split_point])
    test_dict = dict(list(cl.items())[split_point:])
    
    # get dict values
    train_list = list(itertools.chain.from_iterable(train_dict.values()))
    test_list = list(itertools.chain.from_iterable(test_dict.values()))
    
    return train_list, test_list