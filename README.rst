==================
MATS-L1-processing
==================


Code for calibration of MATS images

Description
===========

This package contains two modules, one for level-1 calibration (mats_l1_processing) of MATS data and one for
generating the database to use in this calibration (database_generation).


Installation
==========

1. Install pip in your current envirnoment (e.g. $conda install pip )

1. run $pip install . or $pip install -e . if you want to do development for the package

2. Add calibration_data and testdata folder from box (Calibration/CalibrationSoftware/testdata) 
to root folder. https://chalmersuniversity.box.com/s/d25rklkjtw9shsayff3g34sryru6piv8

3. run pytest by typing "pytest" in root folder

Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.


