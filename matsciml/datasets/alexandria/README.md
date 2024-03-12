Alexandria Database

The [alexandria database](https://alexandria.icams.rub.de/) is maintained by Miguel Marques at RUB.
The database comprises  ~4.5 million three-dimensional relaxed crystal structures that span
the periodic table, in addition to over 100,000 two-dimensional and 10,000 one-dimensional crystal structures.
Further ~400k of the three-dimensional crystal structures are also available at PBEsol geometries and with all properties
calculated with the SCAN functional.
**Warning: The 2D and 1D structures are peridioc in the "non-periodic" directions with a vacuum of 15 Å. Cutoff distances during
graph construction larger than this vacuum will  produce wrong neighborlists.**
Each structures has an associated total energy (eV), forces (eV/Å), band gap (eV), magnetization (Bohr magneton),
magnetic moments on each atom (Bohr magneton),distance to the convex hull per atom (eV/atom),
formation energy per atom (eV/atom) and density of states at the fermi level (states/eV).
During standard processsing of the dataset these quantities are available as training targets.
Stress (eV/Å<sup>2</sup>) data is also available in the database but not added as a target during standard processing.
A fixed training, validation and test split as well as a link to a FAIR repository will be added in the future
with a further publication.

References:

[10.1002/adma.202210788](http://hdl.handle.net/10.1002/adma.202210788) (3D),

[10.1088/2053-1583/accc43](http://hdl.handle.net/10.1088/2053-1583/accc43) (2D),

[10.1126/sciadv.abi7948](http://hdl.handle.net/10.1126/sciadv.abi7948) (method)

[10.1038/s41597-022-01177-w](http://hdl.handle.net/10.1038/s41597-022-01177-w) (PBEsol and SCAN)

Alexandria is available for use under the terms of the Creative Commons Attribution 4.0 License.
Under this license you are free to share and adapt the data, but you must give appropriate credit
to alexandria, provide a link to the license, and indicate if changes were made. You may do so in
any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
