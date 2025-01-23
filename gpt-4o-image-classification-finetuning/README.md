gpt-4o model fine tuning with Azure AI Foundry for image classification

We are using some images from the NEU dataset to predict the class of metal defect.
These defects are categorized into six classes: cracks (cr), inclusions (in), patches (pa), pitted surfaces (ps), rolled-in scale (rs), and scratches (sc).

Step1: We are using Azure Content Safety to detect any issues with the images
Step2: We are going to use the gpt-4o baseline model to predict its class. Accuracy is around 60%
Step3: We are going to fine tune a gpt-4o model with Azure AI Foundry. Accuracy now is around 95%
