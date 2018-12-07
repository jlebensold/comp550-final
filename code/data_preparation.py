from utils import load_struct, save_struct
TRAIN_PER_CAT = 40_000
TEST_PER_CAT = 5_000
def train_test_split():
    records_from_dbpedia = load_struct('data/dbpedia_dump.pickle')
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
    train, test = train_test_split()
    save_struct('data/train_test.pickle', { 'train': train, 'test': test })


