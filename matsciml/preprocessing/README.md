# Preprocessing routines

Currently, there are two main sets of funcitionality for preprocessing
and analysis.

- `atoms_to_graphs.py` is inherited from OCP, and is currently not in use.
- `generate_summary.py`, `produce_split.py` are used for creating compositional splits in
arbitrary datasets.

The latter is used to generate the dataset splits uploaded to Zenodo. At a high
level, we perform uniform random splits in _composition_ space to prevent data
leakage within compositions. Essentially, structures within a composition will
only appear in either training or validation/testing. Furthermore, the training
set is currently guaranteed to include the full set of possible atom types within
a dataset, in that we do not perform validation/testing on atoms that do not
appear in the training set.
