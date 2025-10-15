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

2. run $pip install . or $pip install -e . if you want to do development for the package

3.
	a.  Add calibration_data and testdata folder from box (Calibration/CalibrationSoftware/testdata) 
to root folder. https://su.drive.sunet.se/s/GFLstEoc5R99JKP

	b. This repo contains a subrepo "instrument data". First time cloning the main repo:
		`git submodule update --init --recursive`
	To update the subrepo:
		`git submodule update --recursive --remote`


4. run pytest by typing "pytest" in root folder

Detailed instruction for Windows
==========

1. Download Anaconda navigator and update to newest version

2. Download git for windows

3. Make user account on github.com

4. Make ssh-key in git-bash:
	$ssh-keygen -t ed25519 -C "user@mail.com"
	copy keys to .ssh folder in user home directory
	add config file to .ssh in user home directory
			"Host github.com
			IdentityFile ~.ssh/github"
	test with $ssh -T git@github.com
	add public key to github user preferences

5. Setup user in git-bash:
	$git config --global user.name "UserName"
	$git config --global user.email "email@mail.com"

6. Clone repository
	$git clone git@github.com:innosat-mats/MATS-L1-processing.git

7. Download test and calibraiton data from box and put into root folder of package

8. Setup conda environment
	$conda create -n python=3.9
	$conda install pip

9. Install package
	$pip install -e .

10. Test module
	$pytest


Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
