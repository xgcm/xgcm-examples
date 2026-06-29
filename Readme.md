# xgcm example notebooks

This repository holds the example notebooks for
[xgcm](https://github.com/xgcm/xgcm). They are rendered into the xgcm
documentation at [xgcm.readthedocs.io](https://xgcm.readthedocs.io), which
includes this repository as a git submodule (`docs/xgcm-examples`) pinned to a
specific commit.

The notebooks are stored **with their executed outputs**: the docs site renders
them as-is (`mkdocs-jupyter` with `execute: false`) and does not re-run them. So
when you change a notebook, re-execute it and commit the outputs, then bump the
submodule pointer in the main xgcm repo.

> **History:** this repo previously fed the now-defunct Pangeo Gallery via
> `binderbot`. That pipeline (binder.pangeo.io and its dispatch bot) no longer
> exists, so the gallery integration and its workflows have been removed. The
> notebooks now reach users solely through the xgcm documentation.

## Contributing examples

To contribute an example, fork this repository and add a self-contained
notebook. Provide the data it needs in one of these forms:

1. **Data available in the cloud (preferred).** See `01_eccov4.ipynb`, or
   `03_MOM6.ipynb`, which reads GFDL-CM4 output directly from the analysis-ready
   CMIP6 Zarr store on Google Cloud (read anonymously; needs `zarr`+`gcsfs`).
2. **A small file in a Zenodo archive**, downloaded from within the notebook.
   See `02_mitgcm.ipynb` and `04_nemo_idealized.ipynb`. Use this only for small
   datasets.

After adding or changing a notebook, execute it end-to-end and commit it with
its outputs so the documentation renders correctly.

For more on setting up a development environment, see the
[xgcm contributor guide](https://xgcm.readthedocs.io/en/latest/contributor_guide/).
