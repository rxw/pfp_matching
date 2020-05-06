import face_recognition as fr
import jsonpickle
import json
import os

DBNAME = 'database.json'

def make_encodings(directory):
    filenames = os.listdir(directory)
    if DBNAME in filenames:
        raise Exception('Database already exists for {}'.format(directory))

    full_paths = [os.path.join(directory, fn) for fn in filenames]
    data_struct = {}    


    for path in full_paths:
        face_obj = fr.load_image_file(path)
        encodings = fr.face_encodings(face_obj)

        if len(encodings) > 1:
            print('{} has multiple faces'.format(path))
        elif len(encodings) == 0:
            print('{} has no faces'.format(path))
            continue

        encoding = encodings[0]

        data_struct[path] = jsonpickle.encode(encoding)
    
    dbpath = os.path.join(directory, DBNAME)

    with open(dbpath, 'w') as f:
        f.write(json.dumps(data_struct))

def read_encodings(directory):
    if DBNAME not in os.listdir(directory):
        raise Exception('No database.json to read from in {}'.format(directory))
    
    dbpath = os.path.join(directory, DBNAME)
    db = {}
    with open(dbpath, 'r') as f:
        db_struct = json.loads(f.read())

    for key, val in db_struct.items():
        db[key] = jsonpickle.decode(val)

    return db

