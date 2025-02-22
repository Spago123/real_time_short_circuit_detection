## Real time short circuit predictor

This repo is an example on how to use recurret neural networks for real-time short circuit detection.

Jupyter notebook main.ipynb represent the main point of the repo. In it all the steps needed to train the models are given. In helper_function.py we can find various model generation function that are used.

In folder dataset we have input and output *.csv files that represents the data that is being used to train and the models (we use 80%-20% ratio for training-testing).

In folder figures all the genrated figures are presented that are generated using the plot_data.py script.

In folder models we have the gamma DAE model of the power transmissio line in *.xm format. This models is being used to generate the dataset. For solving the moodel we are using software that can be found on the next link: https://github.com/idzafic/modelSolver.

In folder trained_models we have saved the models that we used in the paper for analysis. (Feel free to playaroud with it and generate more models).

rt_main.py script is the most importat one since this is the file that is actaully simulatin real time dataprocessing. Using this script we are also generating all the files in the plot_data folder (all *.npy files that are there have been generated using this script).

In folder paper you can find the paper n this topic with a detailed explaination about what is being done. Also you can find the *.tex files used for generating the *.pdf paper.


## Contact & Contributing

If you have any questions or would like to contribute to this project, please don't hesitate to contact me via email: harun.spago@gmailcom.
