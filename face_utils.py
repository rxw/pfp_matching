import face_recognition as fr
import os
from PIL import Image

def get_closest_match(faces_folder, face_path):
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
    min_dist = min(face_distances)
    face_distances = face_distances.tolist()
    min_dist_index = face_distances.index(min_dist)

    face_match = valid_images[min_dist_index]
    return face_match
