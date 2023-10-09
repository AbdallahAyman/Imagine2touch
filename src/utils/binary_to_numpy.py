import numpy as np

object_name = "parallelogram"
path = f"/home/abdullah/All_repos_workspace/pseudo_touch/src/data_collection/data/{object_name}/{object_name}_tactile/experiment_1_reskin_ambient"
peek = np.load(path, allow_pickle=True)
print(peek.shape)
