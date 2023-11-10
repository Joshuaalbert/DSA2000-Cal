# FFT Predict

This uses IDG to predict a faint sky model.
IDG requires the gains to be provided in a specific FITS file format, which must match the format that WSClean uses.
The only easy way of making this file is to do a quick dirty image of the DFT visibilities and then modify the FITS file to add extra dimensions.

We then inject these gains into the gains FITS file:
1. ionosphere
2. Antenna pattern
3. instrument effects (J. Lamb note) (TODO)

It then uses the IDG to predict the visibilities.
