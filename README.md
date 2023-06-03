# Summarization-using-Pointer-Generator-Networks

This repository contains the code for Automatic Summarization using Pointer-Generator networks, based on <a href="https://aclanthology.org/P17-1099.pdf">Get To The Point: Summarization with Pointer-Generator Networks</a> implemented in PyTorch framework.

### Framework :
- OS: ```WSL2 Ubuntu 20.04```
- ```Python``` version: 3.8.13
- ```Pytorch``` version: 1.12.1
- GPU used: ```NVIDIA GeForce RTX 2070 Super with Max-Q Design```

### About :
- The dataset used is the <a href="https://huggingface.co/datasets/cnn_dailymail">CNN/DailyMail</a> dataset.
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
    - Average ROUGE-L score is 0.12

- Seq2Seq with Attention & Pointer-Generator Network :
    - Repetition has been decreased although few words / phrases do repeat
    - The words generated still don’t have any relation between them
    - Average ROUGE-L score is 0.34

- Seq2Seq with Attention, Pointer-Generator Network & Coverage Mechanism :
    - Repetition has been decreased further due to the coverage vector
    - The words generated have a relation between them
    - The output is more coherent and makes more sense
    - Average ROUGE-L score is 0.36

### Guidlines to run the code :
- Clone the repository using ```git clone <link_to_the_repo>```
- Using the Conda Package Manager, create a new environment using ```conda create --name <env_name> --file requirements.txt```
- Run the python files using ```python3 <file_name.py>```. For example, to run the baseline model, run ```python3 Seq2Seq_Baseline.py```
- After running, the model will be trained for 20 epochs and the models will get automatically saved.
