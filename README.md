# Disco transformer

This project attempts to produce a generative model capable of
generating DiscoElysium character portrait from facial image of a
person.

## Datasets

The base style [dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) has been downloaded
using torchvision-library. The style dataset has been downloaded from the DiscoElysium wiki. Download
can be replicated by running the `download_disco.sh` script.

## Running the project

1. Download the disco images

        ./download_disco.sh

2. Install depencies (expecting Python 3.10)

        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

3. Add your own images to `assets/my_images` if you wish to see the generative model result

4. Running the training script

        python disco_transformer.py

## Authors

- https://github.com/vahvero