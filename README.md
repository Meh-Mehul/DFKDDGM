# DFKDDGM
A data-free method to distill knowledge of a pre-trained Deep DGM to any student Deep DGM using data-synthesis from Stable Diffusion Model to first re-create the dataset that the teacher DGM was trained on and then distill their knowledge


### Steps to Run
1. First install dependencies from ```requirements.txt```
2. Then train the VAE in ```./VAE``` by following its steps
3. Then run ```bash generate.sh```

#### Note:
    For now, this is hard-coded for CIFAR-10 dataset and a simple VAE as the teacher DGM, but later we will make it modular also
