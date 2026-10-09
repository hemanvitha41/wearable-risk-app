import pickle


def load_subject_raw(subject_id):
    path = f"data/WESAD/{subject_id}/{subject_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")