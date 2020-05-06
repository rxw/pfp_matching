import face_recognition as fr
import os
from PIL import Image

def get_distances(faces_folder, face_path):
    face_image = fr.load_image_file(face_path)
    face_enc = fr.face_encodings(face_image)
    
    # There should only be one face in the photo
    if len(face_enc) > 1:
        raise Exception('There appears to be multiple faces')
    elif len(face_enc) == 0:
        raise Exception('Could not find face in photo')

    face_enc = face_enc[0]
    
    faces_filenames = os.listdir(faces_folder)
    valid_images = []
    valid_img_encs = []

    for ffn in faces_filenames:
        full_path = os.path.join(faces_folder, ffn)
        face_obj = fr.load_image_file(full_path)
        face_encs = fr.face_encodings(face_obj)

        if len(face_encs) == 1:
            valid_images.append((full_path, face_obj))
            valid_img_encs.append(face_encs[0])
    
    # Find the distances and retrieve the closest one
    face_distances = fr.face_distance(valid_img_encs, face_enc)
    final_dists = []

    for i in range(len(face_distances)):
        final_dists.append((valid_images[i][0], valid_images[i][1], face_distances[i]))
    
    return final_dists

def get_closest_match(faces_folder, face_path):
    dist_tups = get_distances(faces_folder, face_path)
    return min(dist_tups, key = lambda t: t[2])

def distance_func(distances):
    s = sum([distance ** 2 for distance in distances])
    return s

def get_closest_to_multiple(faces_folder, faces_path):
    distances = []
    for face_name in os.listdir(faces_path):
        face_path = os.path.join(faces_path, face_name)
        dists = get_distances(faces_folder, face_path)
        distances.append(dists)
    
    distarr_length = len(distances[0])
    possible_indices = [i for i in range(distarr_length)]
    min_index = min(possible_indices, \
            key = lambda n: distance_func([dist[n][2] for dist in distances]))

    return distances[0][min_index]

