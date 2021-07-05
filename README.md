# HILDIF
HILDIF library (InterNLP 2021 Worshop)

## Needed libraries 

* transformers-4.3.3
* faiss-1.5.3
* TextAugment-1.3.4
* contexttimer-0.3.3
* pytorch-pretrained-bert-0.6.2
* fast-influence-functions from https://github.com/salesforce/fast-influence-functions

## Running the debugging pipeline 

augment_without_user should be first run in order to create logs of influential samples for some anchor points. User can then score influential samples for later finetuning using the from_log_retrain_with_user function. 
