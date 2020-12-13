# ctorg-dm
DeepMedic based segmentation on the TCIA CTORG data

## Preprocessing

* body mask creation
search point on skin, region growing, morphological closing, closing/hole filling of top and bottom slice, volumetric hole filling
* splitting of label files into separate masks
* resampling to uniform resolution, identification of bounding box to crop region of interest with margin
* extraction/cropping of region of interest at uniform size across all images
