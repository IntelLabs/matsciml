from matsciml.datasets.alexandria import AlexandriaRequest


# example of downloading the 3D scan dataset
indices = list(range(0, 5))
# The target directory where the LMDB file will be written
lmdb_target_dir = "alexandria_3D_scan"
request = AlexandriaRequest(indices, lmdb_target_dir, dataset="scan")
request.download_and_write(n_jobs=5)

# example of downloading the 3D pbesol dataset
indices = list(range(0, 5))
# The target directory where the LMDB file will be written
lmdb_target_dir = "alexandria_3D_scan"
request = AlexandriaRequest(indices, lmdb_target_dir, dataset="pbesol")
request.download_and_write(n_jobs=5)

# example of downloading the 3D pbe dataset
indices = list(range(0, 45))
# The target directory where the LMDB file will be written
lmdb_target_dir = "alexandria_3D_pbe"
request = AlexandriaRequest(indices, lmdb_target_dir, dataset="pbe")
request.download_and_write(n_jobs=5)

# example of downloading the 2D pbe dataset
indices = list(range(0, 2))
# The target directory where the LMDB file will be written
lmdb_target_dir = "alexandria_2D"
request = AlexandriaRequest(indices, lmdb_target_dir, dataset="2D")
request.download_and_write(n_jobs=2)

# example of downloading the 1D pbe dataset
indices = list(range(0, 1))
# The target directory where the LMDB file will be written
lmdb_target_dir = "alexandria_1D"
request = AlexandriaRequest(indices, lmdb_target_dir, dataset="1D")
request.download_and_write(n_jobs=1)
