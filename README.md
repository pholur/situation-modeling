# NP2IO - Threat Detection in Noun Phrases 
*Insider-Outsider Classification - A new paradigm for Identifying Conspiracy Theories*

## Pointers to get started:
1. `imports.py`: Change the variables here to access the model checkpoints, dataset, and intermediate `pkl` files available on [OSF](https://osf.io/hgnm7/). Pay particular attention to `*DATA_PATH` as well as the `OPT` option to switch between training and inference.
2. Fine-tuning NP2IO is performed on 1-2 Titan RTX GPUs for about 2-3 hours with one of the GPUs solely attributed to data augmentation via BERT. Set `REEXTRACT` and `SAVE` to `False` to use a single-GPU for training post-data augmentation. The post-data augmented input files are available on [OSF](https://osf.io/hgnm7/).
3. To train, run `python train.py`. TODO: Dockerize the environment. Till then transformers, pytorch1.7 is sufficient with additional minor dependencies.
4. To test, run `python SCRIPT_test_against_baselines.py`. This call takes a while - currently does not generate real-time outputs so it is a little annoying to run.
5. `test_inline.ipynb` provides a table-top setup for qualitative eval.
6. Other files are helpers.

Contact [me](pholur@g.ucla.edu) if you have any questions.
