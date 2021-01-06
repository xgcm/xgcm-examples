[![](https://img.shields.io/website?down_message=down&label=pangeo%20gallery&up_message=up&url=https%3A%2F%2Fgallery.pangeo.io%2Frepos%2Fxgcm%2Fxgcm-examples%2F)](https://gallery.pangeo.io/repos/xgcm/xgcm-examples/)

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
2. Add files to the [xgcm-examples zenodo archive](https://zenodo.org/record/4421428#.X_XP7y1h3x9) and access them from within the notebook *(see `02_mitgcm.ipynb` and `04_nemo_idealized.ipynb` for examples). This should only be done for small datasets.
3. Other way of accessing data online and download locally from the notebook: See `03_MOM6.ipynb` for examples.
