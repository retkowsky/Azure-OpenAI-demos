# gpt-4o model fine tuning with Azure AI Foundry for image classification

We are using some images from the NEU dataset to predict the class of metal defect.
These defects are categorized into six classes: cracks (cr), inclusions (in), patches (pa), pitted surfaces (ps), rolled-in scale (rs), and scratches (sc).

- <a href=" ">Go to Notebook 1</a> We are using Azure Content Safety to detect any issues with the images
- Step2: We are going to use the gpt-4o baseline model to predict its class. Accuracy is around 60%
- Step3: We are going to fine tune a gpt-4o model with Azure AI Foundry. Accuracy now is around 95%

## 1 Model training
<img src="capture1.jpg">

## 2 Confusion matrix of the gpt-4o baseline model
<img src = "baseline_confmatrix.png">

## 3 Confusion matrix of the fine-tuned gpt-4o model
<img src = "ft_confmatrix.png">

## 4 Model is trained
<img src="capture2.jpg">

## 5 Deploying the model
<img src="capture3.jpg">

## 6 Model is deployed
<img src="capture4.jpg">

## 7 We can use the deployed model
<img src="capture5.jpg">
