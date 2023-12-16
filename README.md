# Classifying_Galaxies
### Collaborators: Caleb Painter, John DiNovi, Adam Boesky, and Santiago Calderon
*This work was done as the final project for the CS109a: Introduction to Data Science course at Harvard University.*


## Summary
A machine learning study on galaxy Hubble class and morphology. We evaluate the classification performance of a number of different models at classifying a dataset from [Karachentsev et al. 2013](https://ui.adsabs.harvard.edu/abs/2013AJ....145..101K/abstract). We find that the best model for classifying galaxies is XGBoost, and with this classifier we analyze the feature importance of the galaxy properties. We also offer our own model, which struggles to classify the TT morphology accurately---we hypothesize that the poor performance of the custom model is a result of small, and imbalanced nature of the dataset.


## Repository Structure
The following is a description of the repository structure:

**`Data/`** ---   The data that we use for this study.

**`eda/`** --- Preliminary data analysis that we conducted.

**`milestones/`** --- Reports on the different milestones of the study. Contains the **final notebook**, as well as the **final video submission**.

**`modeling/`** --- Inital modeling. Contains the custom model in `modeling-utils.py`.

**`extra/` & `other/`** --- Random stuff.


## Dataset
The following is a description of the fields in the dataset we use:

**Response Variables:**

    TT: Morphological type --> This is the Hubble stage of a galaxy, which can be grouped into appropriate Hubble classes (Spiral, Lenticular, Irregular)

    Mcl: Dwarf galaxy morphology --> Some galaxies in the sample are dwarf galaxies and can be further classified as to what kind of dwarf they are

**Predictors:**

    Tdw: Dwarf galaxy surface brightness morphology

    FUV: Far UV

    Bmag: Apparent integral b-band magnitude

    Hamag: Integral H-alpha line emission magnitude

    Kmag: K-band magnitude

    HImag: HI 21cm line magnitude

    W50: HI Line width at 50% level from maximum

    HRV: Heliocentric radial velocity

    Dist: linear distance to galaxy from the sun (heiocentric)

    A26: major linear diameter

    i: inclination of galaxy from the face on

    vAmp: amplitude of rotational velocity

    Bmu: average b-band surface brightness

    M26: log-mass within the Holmberg radius

    MHI: log-Hydrogen mass

    Vlg: local group radial velocity

    Ti5: tidal index
