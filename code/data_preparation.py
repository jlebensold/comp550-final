from utils import load_struct, save_struct
TRAIN_PER_CAT = 40_000
TEST_PER_CAT = 5_000
def train_test_split():
    """ Split dbpedia pickle into training / test as per the original paper """
    records_from_dbpedia = load_struct('../data/dbpedia_dump.pickle')
    counts = Counter([ rec['category'] for rec in records_from_dbpedia])
    categories = list(counts.keys())

    test_dataset = []
    train_dataset = []
    for category in categories:
        subset = [ rec for rec in records_from_dbpedia if rec['category'] == category]
        indexes = set(np.arange(0, len(subset)))
        train_indexes = np.random.choice(list(indexes), TRAIN_PER_CAT, replace=False)
        test_indexes = np.random.choice(list(indexes - set(train_indexes)), TEST_PER_CAT, replace=False)
        for idx in train_indexes:
            train_dataset.append(subset[idx])
        for idx in test_indexes:
            test_dataset.append(subset[idx])
    return train_dataset, test_dataset

def pickle_train_test():
    """ Save training and test set """
    train, test = train_test_split()
    save_struct('../data/train_test.pickle', { 'train': train, 'test': test })


def extract_from_dbpedia():
    """ paginate through requests to DbPedia to extract abstracts based on
    category"""

    query = """
    PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?iri ?abstract {{
    ?iri a dbpedia-owl:{0} ;
        dbpedia-owl:abstract ?abstract ;
        rdfs:label ?lbl .
    FILTER( langMatches(lang(?abstract),"en") )
    }}
    LIMIT {1} OFFSET {2}
    """
    import numpy as np
    queries = [
    #    ('Company', 63_058),
    #    ('EducationalInstitution', 50_450),
    #    ('Artist', 95_505),
    #    ('Athlete', 268_104),
    #    ('OfficeHolder', 47_417),
    #    ('MeanOfTransportation', 47_473),
    #    ('Building', 67_788),
    #    ('NaturalPlace', 60_091),
    #    ('Village', 159_977),
    #    ('Animal', 180_000),
    #    ('Plant', 50_585),
    #    ('Album', 117_683),
    #    ('Film', 86_486),
    #    ('WrittenWork', 55_174),

    ]
    page_size = 10_000
    dataset = []
    for (class_type, limit) in queries:

        full_pages = int(limit / page_size)
        last_page_size = limit % page_size
        for idx, cur_page in enumerate(np.arange(0, full_pages + 1)):
            offset = cur_page * page_size

            query_page_size = page_size
            if (idx == full_pages):
                query_page_size = last_page_size



            sparql.setQuery(query.format(class_type, query_page_size, offset))
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            print("querying... ", class_type, limit, len(results["results"]["bindings"]))
            for result in results["results"]["bindings"]:
                dataset.append({
                    'uri': result['iri']['value'],
                    'abstract': result['abstract']['value'],
                    'category': class_type
                })

