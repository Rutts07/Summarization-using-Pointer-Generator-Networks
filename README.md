# Summarization-using-Pointer-Generator-Networks

This repository contains the code for Automatic Summarization using Pointer-Generator networks, based on <a href="https://aclanthology.org/P17-1099.pdf">Get To The Point: Summarization with Pointer-Generator Networks</a> implemented in PyTorch framework.

### Framework :
- OS: ```WSL2 Ubuntu 20.04```
- ```Python``` version: 3.8.13
- ```Pytorch``` version: 1.12.1
- GPU used: ```NVIDIA GeForce RTX 2070 Super with Max-Q Design```

### About :
- The dataset used is the <a href="https://www.tensorflow.org/datasets/catalog/cnn_dailymail">CNN/DailyMail</a> dataset.
- A total of 3 model architectures have been implemented:
    - A vanilla Sequence-to-Sequence model with Attention (Baseline)
    - A Sequence-to-Sequence model with Attention and Copy Mechanism (Pointer-Generator Network)
    - A Sequence-to-Sequence model with Attention, Copy Mechanism and Coverage Mechanism (Pointer-Generator Network with Coverage Mechanism)
- All the models have been trained on the CNN/DailyMail dataset (30K sentences) for 20 epochs.
- The models have been evaluated on the test set (1K sentences) using the ROUGE-L metric.

### Observations & Results :
- Seq2Seq with Attention Baseline :
    - Repetition of the same word is evident from the output
    - The output is doesn’t make any sense both syntactically and semantically

- Seq2Seq with Attention & Pointer-Generator Network :
    - Repetition has been decreased although few words / phrases do repeat
    - The words generated still don’t have any relation between them

- Seq2Seq with Attention, Pointer-Generator Network & Coverage Mechanism :
    - Repetition has been decreased further
    - The words generated have a relation between them
    - The output is more coherent and makes more sense

### Guidlines to contribute to the project : 
- Clone the repository using ```git clone <link_to_the_repo>```
- Using the Conda Package Manager, create a new environment using ```conda create --name <env_name> --file requirements.txt```
- Create a New Branch from the latest commit to push any changes.
    - ```git checkout main``` to change to the main (default) branch.
    - ```git pull origin main``` to pull the latest changes.
    - ```git checkout -b <branch_name>``` to create and shift to the new branch.
    - After making changes in the new branch, ```git add .``` and ```git commit -m "Your Message"``` them.
    - ```git push origin <branch_name>```
- Start a Pull Request to merge the new branch to the existing branch. Wait for the approval of a fellow collaborator.
- Merging to be done by another collaborator only by resolving conflicts if any.
- After your PR is merged, delete the branch.
