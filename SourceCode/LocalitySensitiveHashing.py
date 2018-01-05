#Importing Dependencies
import numpy as np
import glob
from PIL import Image
from pyspark import SparkConf, SparkContext
from pyspark import StorageLevel
from bitarray import bitarray
import sympy
import sys
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#to store images and names
images = []
names = []


#Loading images from different folders
image_filenames = glob.glob("imagenet_object_detection_test/ILSVRC/Data/DET/test/*.JPEG")
for image_file in image_filenames[:2000]:
    try:
        image_data = np.asarray(Image.open(image_file).convert("RGB"))[:, :, :3]
        images.append(image_data)
        names.append(image_file)
    except:
        pass

image_filenames = glob.glob("imagenet_object_localization/ILSVRC/Data/CLS-LOC/val/*.JPEG")
for image_file in image_filenames[:2000]:
    try:
        image_data = np.asarray(Image.open(image_file).convert("RGB"))[:, :, :3]
        images.append(image_data)
        names.append(image_file)
    except:
        pass

image_filenames = glob.glob("imagenet_object_detection_test/ILSVRC/Data/DET/train/*/*.JPEG")
for image_file in image_filenames[:10000]:
    try:
        image_data = np.asarray(Image.open(image_file).convert("RGB"))[:, :, :3]
        images.append(image_data)
        names.append(image_file)
    except:
        pass

conf = SparkConf().setMaster("spark://ip-172-31-46-196.ec2.internal:7077").setAppName("LSH").set('spark.executor.memory', '32G').set('spark.driver.memory', '32G').set('spark.driver.maxResultSize', '8G')
sc = SparkContext(conf=conf)

rdd1 = sc.parallelize(zip(names,images), 64)

#shingle size
k = 16

#SHINGLING
def map1(tuple1):
    name, image = tuple1
    shingles = []
    for i in range(image.shape[0] - k):
        for j in range(image.shape[1] - k):
            red_img = image[i:i+k, j:j+k, :].flatten()
            shingles.append(tuple(red_img))
    return (name, shingles)

rdd2 = rdd1.map(map1)

rdd2.cache()

rdd3 = rdd2.flatMap(lambda r: r[1])

k_shingle_space = rdd2.distinct().collect()

N = len(k_shingle_space)

def map2(tuple1):
    name, image = tuple1
    bool_vec = bitarray(len(k_shingle_space))
    bool_vec.setall(0)
    for i in range(len(k_shingle_space)):
        if k_shingle_space[i] in image:
            bool_vec[i] = 1 
    return (name, bool_vec)

#MIN HASHING
sig_size = 100

a = np.random.randint(low = -100000, high = 100000, size = sig_size)
b = np.random.randint(low = -100000, high = 100000, size = sig_size)

for k in range(N):
    if sympy.isprime(k+N+1):
        p = k+N+1
        break

MAX = sys.maxsize

def map3(tuple1):
    name, image = tuple1
    M = [MAX]*sig_size
    for i in range(N):
        if image[i]:
            for j in range(sig_size):
                h = ((a[j]*(i+1) + b[j])%p)%N
                if h < M[j]:
                    M[j] = h
    return (name, M)

rdd4 = rdd2.map(map2).map(map3)

#LOCALITY SENSITIVE HASHING
d = {} #capture candidates that are hashed to same bucket (identical band)
bands = 20
r = 5

sig_matrix = rdd4.collect()

for name, image in sig_matrix:
    for i in range(bands):
        band = tuple(image[i*r:(i+1)*r])
        if band not in d:
            d[band] = [name]
        else:
            d[band].append(name)

for band in d:
    if len(d[band]) > 1 and len(d[band]) < 100: #for bands that are common for almost all images
        print(band, d[band])

sig_dict = dict(sig_matrix)

def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

jaccard_candidates = {}
for band in d:
    if len(d[band]) > 1 and len(d2[band]) < 100:
        for i in range(len(d[band])):
            for j in range(i+1, len(d[band])):
                if i != j:
                    score = jaccard_similarity_score(sig_dict[d[band][i]], sig_dict[d[band][j]])
                    if score > 0.8:
                        if d[band][i] not in jaccard_candidates:
                            jaccard_candidates[d[band][i]] = [d[band][j]]
                        else:
                            if d[band][j] not in jaccard_candidates[d[band][i]]:
                                jaccard_candidates[d[band][i]].append(d[band][j])

for name1 in jaccard_candidates:
    candidates = []
    image_data = np.asarray(Image.open(name1).convert("RGB"))[:, :, :3]
    candidates.append(image_data)
    for name in dummy[name1]:
        image_data = np.asarray(Image.open(name).convert("RGB"))[:, :, :3]
        candidates.append(image_data)
    show_images(candidates)
    break #comment this if you want to see all the groups

#RANDOM HYPERPLANES
sk_size = 100

rnd_hyp_plns = []
for _ in range(sk_size):
    rnd_hyp_pln = np.random.randint(low = -1, high = 1, size = N) #it actually gives a vector of 0's and -1's
    for i in range(N):
        if ran_hyp_pln[i] == 0:
            ran_hyp_pln[i] = 1
    rnd_hyp_plns.append(rnd_hyp_pln)

def map4(tuple1):
    name, image = tuple1
    sketch_vec = [-1]*sk_size
    for j in range(sk_size):
        hv = np.array(image).dot(np.array(rnd_hyp_plns[j]))
        if hv >= 0:
            sketch_vec[j] = 1
    return (name, sketch_vec)

rdd5 = rdd2.map(map2).map(map4)

sketch_matrix = rdd5.collect()

d2 = {} #capture candidates that are hashed to same bucket (identical band)

for name, image in sketch_matrix:
    for i in range(bands):
        band = tuple(image[i*r:(i+1)*r])
        if band not in d2:
            d2[band] = [name]
        else:
            d2[band].append(name)

for band in d2:
    if len(d[band]) > 1 and len(d[band]) < 100: #for bands that are common for almost all images
        print(band, d[band])

sketch_dict = dict(sketch_matrix)

cosine_candidates = {}
for band in d:
    if len(d2[band]) > 1 and len(d2[band]) < 100:
        for i in range(len(d[band])):
            for j in range(i+1, len(d[band])):
                if i != j:
                    score = cosine_similarity(sketch_dict[d[band][i]], sketch_dict[d[band][j]])[0][0]
                    if score > 0.8:
                        if d[band][i] not in cosine_candidates:
                            cosine_candidates[d[band][i]] = [d[band][j]]
                        else:
                            if d[band][j] not in cosine_candidates[d[band][i]]:
                                cosine_candidates[d[band][i]].append(d[band][j])

for name1 in cosine_candidates:
    candidates = []
    image_data = np.asarray(Image.open(name1).convert("RGB"))[:, :, :3]
    candidates.append(image_data)
    for name in dummy[name1]:
        image_data = np.asarray(Image.open(name).convert("RGB"))[:, :, :3]
        candidates.append(image_data)
    show_images(candidates)
    break #comment this if you want to see all the groups