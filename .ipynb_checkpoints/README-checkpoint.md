# PrIMP (Prediction of ion-channel modulating peptides)

PrIMP predicts the modulability of calcium, sodium, potassium ion channels as well as nicotine acetylcholine receptors (nAChRs) by accurate prediction models from peptide sequences only.

<br>

## Dependencies
To install python dependencies, run: `pip install -r requirements.txt`

<br>

## Input file 
Input file type is fasta format in which amino acids are represented using single-letter codes.

Limitation of each sequence length is <=300.

Detailed information of fasta links: https://en.wikipedia.org/wiki/FASTA_format

<br>

## Output file
Output file contains information about probabilities of four respective ion-channel modulability.

Probability with >=0.5 predicts as modulator peptides for respective ion channels.

<br>

## Running PrIMP
`python PrIMP.py --fasta <input_fasta_file.fasta> --output <output_file_name.csv>`

Example fasta file is provided. To check whether the PrIMP is run, please run the command

`python PrIMP.py --fasta example.fasta --output results.csv`
<br>

## Reproducibility of tranining PrIMP
`
python primp_training.py --tgtList <tgtList> --valFold <valFold> --modelID <modelID> --TL <TL> --c <Dataset sequence identity>
`
- Example
`
python primp_training.py --tgtList all --valFold 1 --modelID 3 --TL True --c 0.8
`

> ### tgtList
> 
> There are four targets and the datset are in ./Data folder. tgtList can be one of five options
> - calcium
> - potassiu
> - nAChRs
> - sodium
> - all <br>
> If all is chosen, the four targets were used simultaneously for model traning by multi-task learning.

> ### valFold
> The datasets were prepared for 5-fold cross-validation, thus the training datasets were prepared by splitting into 5-folds.
> valFold assigned the number of fold for the validation dataset during model training. 
> Enter a value from 0 to 4.

> ### modelID
> There are 16 model architecture parameters in the code, the desired model architectures are automatically generated.
> Enter a value from 0 to 15.

> ### TL
> To apply transfer learning, True, if not, False.
> Pre-trained weights were prepared by training the model with antimicrobial peptide dataset.

> ### c
> There were four train/test datasets for respective target ion channels, and the datasets were filtered with varying sequence identity parameter for CD-HIT.
> To reduce the sequence identity of training dataset, the sequence identity cut-offs of 0.7, 0.8, 0.9, and 0.95 were used. 
> As increasing the sequence identity, there are possibly more similar sequences with increasing the amount of the training dataset.
> Enter one of 0.7, 0.8, 0.9, and 0.95.


<br>

## Dataset used for training PrIMP
There are dataset for training and evaluate model performance for PrIMP in /Data folder.

<br>

## Pretrained weights with AMP dataset
The pretrained weights for transfer learning are provided in /modelWeights/DBAASP

<br>

## Citation

