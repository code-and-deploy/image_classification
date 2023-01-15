# Build and Deploy an Image Classification Model

This tutorial consists of 3 major steps:

1- Generate Images using Text for classes of interest

2- Train a basic ResNet model on the generated images

3- Deploy the trained ResNet model on Gradio and test it on Images from the Internet


## Steps to run this code

### 0. Install Python Virtual Environment

### 1. Generate the dataset

### 2. Train & Validate the model

Run the following command

```python
python trainval.py -e baselines -sb ./results -r 1
```

Argument Descriptions:
```
-e  [Experiment group to run like `baselines`] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
```


### 3. Visualize the Results

Open `results.ipynb` and visualize the results as follows

<p align="center" width="100%">
<img width="100%" src="https://raw.githubusercontent.com/haven-ai/haven-ai/master/docs/vis.gif">
</p>



### 4. Deploy the model on Gradio



### Contact me

If you would like to chat or collaborate, you can reach me at [twitter](https://twitter.com/ILaradji) or [LinkedIn](https://www.linkedin.com/in/issam-laradji-67ba1a99/).
