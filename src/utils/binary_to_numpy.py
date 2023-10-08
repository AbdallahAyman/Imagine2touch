import numpy as np

path = "/export/home/ayada/reskin_ws/src/touch2image/reskin/data_collection/data/square/square_tactile/experiment_6_reskin"
peek = np.load(path, allow_pickle=True)
print(peek.shape)
