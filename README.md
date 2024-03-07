[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13011758&assignment_repo_type=AssignmentRepo)

# Important Note

- Each of our jupyter notebook takes about an hour to run due to
    - the function to choose number of clusters (pairwiseCrossValidation) takes about 20 minutes to run
    - the function to run the alignment algorithm (runClustalInRange) takes about 30 minutes to run

- DNABERT6 and Finetuned_DNABERT6 takes about 4-6 hours to get the embeddings

For these reasons we provide all the necessary files to get our results, in this drive link:
https://drive.google.com/drive/folders/1KLiCzlLoEf0avWA5S5f40Lqq-E_cfNEX?usp=sharing

# Repository Organization

- run.ipynb : this is the model with the best results, the training loop is commented, you can just load the .pth file. 
    - We recommend not to run "runClustalInRange" function, you can find the required files in our drive folder mentioned above.

- helper.py : we have all our utility functions in this file, all notebook import this file

- plot_clusters.py : this function was provided to us by Antoine Tappy who is working on a similar project

- data_preprosessing.ipynb : we did all our data preprocessing in this file, you can find all the data we used in Data.zip

- Other_Notebook : as we tried many different models, we keep all of our other notebooks here, there are some duplicate files such as helper.py, to simplify importing

# Requirements

- clustal
- biopython
- numpy
- sklearn
- scipy
- pytorch
- matplotlib

# Data & Pretrained Models:

## Data

We got our data from [NCBI's website](https://www.ncbi.nlm.nih.gov/nuccore)

The procedure is as follows
- Enter rbcL
- Filter for plants only
- Select the sequence length 
    - rbcL : 600 to 1000 -> ~90k Sequence, ~80Mo
- Download it to have the information and the DNA sequence Click and send to (corner top right) > Complete Record > File > Format = Fasta > Sort by Taxonomy ID
- Put the fasta file into /Data

## Pretrained models

- We used 2 pretrained models
    - We used models from [DNABERT](https://github.com/jerryji1993/DNABERT)
        - [Pretrained DNABERT6](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing)
        - [FineTuned DNABERT6](https://drive.google.com/drive/folders/15wFcukTv3ecPw9_25dcOv-bZmj-8d_-6?usp=sharing)

- We also provide our trained models as .pth files in our repository

You can find these models in our drive folder as well

# Running run.ipynb
 
- To be able to run run.ipynb, please extract Data.zip with the same name in the same directory as run.ipynb
- As mentioned before in "Important Note" we highly recommend downloading /clustal /clusters and /plots folders from our drive to avoid running the notebook for an hour