from datasketch import MinHash, MinHashLSH
from PIL import Image
import os
import numpy as np
import shutil
import time

def get_minhash(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((16, 16))
    img_array = np.array(img).flatten()
    m = MinHash(num_perm=128)
    for d in img_array:
        m.update(d.tobytes())
    return m

def index_images(folder):
    image_minhashes = {}
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        minhash = get_minhash(filepath)
        lsh.insert(filename, minhash)
        image_minhashes[filename] = minhash
    return image_minhashes

start_time = time.time()

lsh = MinHashLSH(threshold=0.9, num_perm=128) 

folder1 = 'imageset\\test'
folder2 = 'imageset\\original'
duplicates_folder = 'imageset\\duplicates'
os.makedirs(duplicates_folder, exist_ok=True)

images1_minhashes = index_images(folder1)
images2_minhashes = index_images(folder2)

with open('LSH_result.txt', 'w') as file:
    for filename, minhash in images1_minhashes.items():
        duplicates = lsh.query(minhash)
        if len(duplicates) > 1:
            file.write(f"Near-duplicates for {duplicates}\n")
            
            src_path = os.path.join(folder1, filename)
            dest_path = os.path.join(duplicates_folder, filename)
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)

end_time = time.time()
total_time = end_time - start_time

with open('LSH_result.txt', 'a') as file:
    file.write(f"Execution Time: {total_time} seconds\n")