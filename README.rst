==================
MATS-L1-processing
==================


Code for calibration of MATS images

Deployments status
==================

.. image:: https://zenodo.org/badge/176311526.svg
   :target: https://zenodo.org/badge/latestdoi/176311526
   :alt: DOI

.. image:: https://github.com/innosat-mats/level1a/actions/workflows/ci.yml/badge.svg
   :alt: CI

.. image:: https://github.com/innosat-mats/level1a/actions/workflows/cd.yml/badge.svg
   :alt: CD

Description
===========

This package contains two modules, one for level-1 calibration (mats_l1_processing) of MATS data and one for
generating the database to use in this calibration (database_generation).


Installation
==========

1. Choose one of the supported environment workflows:

	a. ``uv`` workflow (recommended for CI and pip users):

		- Create environment and install development dependencies:
		  ``uv sync --group dev``
		- Install package in editable mode:
		  ``uv pip install -e .``

	b. Conda workflow (recommended for research users):

		- Create and activate environment:
		  ``conda env create -f deps/research-conda/environment.yml``
		  ``conda activate mats-l1-processing``
		- Keep editable install from the repo root:
		  ``pip install -e .``

2. Dependency tracks are split to reduce deployment drift:

	- ``deps/research-conda/environment.yml``: researcher environment
	- ``deps/infra-cdk/requirements.in``: CDK/deployment tooling
	- ``deps/lambda-runtime/requirements.in``: Lambda runtime dependencies
	- ``deps/shared/base.in``: shared baseline constraints

3. Maintain lock files for Lambda and infrastructure with ``uv``:

	- ``uv pip compile -p 3.11 -o l1b_lambda/requirements.txt l1b_lambda/requirements.in``
	- ``uv pip compile -p 3.11 -o l1b_lambda/level1b/requirements.txt l1b_lambda/level1b/requirements.in``

4.
	a.  Add calibration_data and testdata folder from box (Calibration/CalibrationSoftware/softwaredata) 
to root folder. https://su.drive.sunet.se/s/6NiSdeYL7yPFdiX

	b. This repo contains a subrepo "instrument data". First time cloning the main repo:
		`git submodule update --init --recursive`
	To update the subrepo:
		`git submodule update --recursive --remote`


5. run pytest by typing "pytest" in root folder

Replay Level1A files to Level1B queue
==========

Use ``scripts/enqueue_level1b_sqs.py`` to enqueue synthetic S3 notifications to the Level1B SQS queue.

- Preview what would be sent:
	``uv run --with boto3 scripts/enqueue_level1b_sqs.py --start 2023-02-01 --end 2023-05-10 --dry-run``
- Preview using hourly partitions (YYYY/MM/DD/HH) under a base prefix:
	``uv run --with boto3 scripts/enqueue_level1b_sqs.py --start 2023-02-01 --end 2023-02-02 --dry-run --profile mats --prefix CCD``
- Send messages for CCD default ops bucket/queue:
	``uv run --with boto3 scripts/enqueue_level1b_sqs.py --start 2023-02-01 --end 2023-05-10``
- Use development queue defaults:
	``uv run --with boto3 scripts/enqueue_level1b_sqs.py --start 2023-02-01 --end 2023-05-10 --development``
- Use a specific AWS profile:
	``uv run --with boto3 scripts/enqueue_level1b_sqs.py --start 2023-02-01 --end 2023-05-10 --dry-run --profile <your-profile>``

Useful flags:

- ``--data-source PM`` for photometer track
- ``--bucket`` and ``--queue-name`` to override defaults
- ``--prefix`` is the base path before partition folders ``YYYY/MM/DD/HH``
- ``--match-mode`` can be ``partition`` (default) or ``last-modified``
- ``--max-messages`` to cap replay size

Detailed instruction for Windows
==========

1. Install one of the supported environment managers:

	a. Recommended for researchers (Conda):
		Install Anaconda or Miniconda.

	b. Recommended for CI-style local workflows (uv):
		Install ``uv`` from https://docs.astral.sh/uv/getting-started/installation/

2. Download Git for Windows.

3. Make a user account on GitHub.

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

7. Download test and calibration data from box and put into root folder of package.

8. Set up environment (choose one):

	a. Conda workflow:
		$conda env create -f deps/research-conda/environment.yml
		$conda activate mats-l1-processing

	b. uv workflow:
		$uv sync --python 3.11 --group dev

9. Install package
	$pip install -e .
	or
	$uv pip install -e .

10. Test module
	$pytest
	or
	$uv run pytest
