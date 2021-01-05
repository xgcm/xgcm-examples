# XGCM Example Gallery (under construction)

This repository stores the xgcm example repo for the Pangeo Gallery.
It is configured to automatically build itself using
[binderbot](https://github.com/pangeo-gallery/binderbot).
It is linked, via a git submodule, the the
[gallery website repo](https://github.com/pangeo-gallery/pangeo-gallery).
Whenever the notebooks are updated in this, repository
dispatch is used to trigger a gallery rebuild. This keeps
[gallery.pangeo.io](http://gallery.pangeo.io) always in sync with this repo.

The repo contains the following elements:

- A set of jupyter notebooks, numbered in the order that we want them to
  appear on the gallery website.
- A configuration file, `binder-gallery.yaml`, which provides important
  configuration parameters (see [pangeo gallery documentation](http://gallery.pangeo.io)).
- A thumbnail image (`thumbnail.png`), a 200 x 200 px image which represents
  the gallery content.
- Github workflows, which make the magic happen! (Don't touch these.)

## Contributing Examples

To contribute examples, please fork this repository and add a self contained notebook. The data for the example should be provided in one of these forms:

1. Data available in the cloud (preferred): See `01_eccov4.ipynb` for example.
2. Other way of accessing data online and download locally from the notebook: See `03_MOM6.ipynb` and `04_nemo_idealized.ipynb` for examples.
3. For very small examples, it is acceptable to commit the files directly to the `datasets` folder: See `02_mitgcm.ipynb` for example.
