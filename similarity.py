import os
from os.path import join
import pickle
import pprint
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import resampy
import seaborn as sns
from scipy.io.wavfile import read
from tensorflow.keras import Model

import params
import yamnet
from features import waveform_to_log_mel_spectrogram_patches

p = params.Params()

def prepare_wav(path):
    sr, wav = read(path)

    if len(wav.shape) > 1:
        wav = wav.mean(axis=1) / 2**15  # convert to mono
    if sr != p.sample_rate:
        wav = resampy.resample(wav, sr, p.sample_rate)
    wav = wav.astype("float32")

    # _, feats = waveform_to_log_mel_spectrogram_patches(wav, p)

    return wav


# def load_similarity_function():

#     model = yamnet.yamnet_frames_model(p)
#     model.load_weights("/Users/maxs/Downloads/yamnet.h5")

#     compute_embedding = Model(model.inputs, model.layers[-3].output)

#     def similarity(wav1_path, wav2_path):
#         wav1 = prepare_wav(wav1_path)
#         wav2 = prepare_wav(wav2_path)

#         embed1 = compute_embedding(wav1)
#         embed2 = compute_embedding(wav2)

#         # yamnet processes sound in chunks and returns a row for each chunk use
#         # the minimum number of chunks present for each sound (i.e. truncate
#         # the tensor) to avoid array size issues
#         nrows = min(embed1.shape[0], embed2.shape[0])
#         embed1 = embed1[:nrows, :]
#         embed2 = embed2[:nrows, :]

#         dist = np.linalg.norm(embed1 - embed2)

#         return dist

#     return similarity

model = yamnet.yamnet_frames_model(p)
model.load_weights("yamnet.h5")

global compute_embedding
compute_embedding = Model(model.inputs, model.layers[-3].output)

def compute_similarity(wav1_path, wav2_path):
    print("starting " + wav1_path + " and " + wav2_path)
    global compute_embedding
    wav1 = prepare_wav(wav1_path)
    wav2 = prepare_wav(wav2_path)

    embed1 = compute_embedding(wav1)
    embed2 = compute_embedding(wav2)

    # yamnet processes sound in chunks and returns a row for each chunk use
    # the minimum number of chunks present for each sound (i.e. truncate
    # the tensor) to avoid array size issues
    nrows = min(embed1.shape[0], embed2.shape[0])
    embed1 = embed1[:nrows, :]
    embed2 = embed2[:nrows, :]

    dist = np.linalg.norm(embed1 - embed2)

    sound1_name = wav1_path.split('/')[-1]
    sound2_name = wav2_path.split('/')[-1]
    return sound1_name, sound2_name, dist

def recompute():

    base_path = "/Users/maxs/Downloads/edited_stims/"
    files = [join(base_path, f) for f in os.listdir(base_path) if not f.endswith("mp4")]

    # remove "2" versions
    files = [f for f in files if '2' not in f]

    from multiprocessing import Pool
    p = Pool(9)

    todo = list(combinations(files, 2))
    sims = p.starmap(compute_similarity, todo)

    sim_dict = {(k1, k2): k3 for k1, k2, k3 in sims}

    save_computed_similarity(sim_dict)
    pprint.pprint(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True))

    return sim_dict



def plot():
    sims = load_computed_similarity()

    sims_list = [(k1, k2, val) for ((k1, k2), val) in sims.items()]
    scores = pd.DataFrame(sims_list, columns=['Sound 1', 'Sound 2', 'score'])
    # truncate
    scores['Sound 1'] = scores['Sound 1'].str.slice(stop = -4)
    scores['Sound 2'] = scores['Sound 2'].str.slice(stop = -4)
    scores = scores.pivot(index = "Sound 1", columns = "Sound 2")
    # f = plt.figure(figsize=(15, 15))
    sns.set_theme(font_scale = .75)
    f = sns.heatmap(scores, yticklabels=True, xticklabels=True)

    f.figure.tight_layout()
    # f.ax.subplot()

    # plt.subplots_adjust(left=.5, bottom=.5)
    plt.show()

def save_computed_similarity(sim_dict):
    with open('computed_similarities.pkl', 'wb') as f:
        pickle.dump(sim_dict, f)

def load_computed_similarity():
    with open('computed_similarities.pkl', 'rb') as f:
        sim_dict = pickle.load(f)
    return sim_dict

if __name__ == "__main__":

    # this line fixes an error that makes running using ipython difficult
    __spec__ = None

    sim_dict = recompute()
    sims = load_computed_similarity()

    plot()
    import pdb; pdb.set_trace()
