# Generative Classification
Here Gaussian Mixture Model (GMM) is used as a generative classifier.


A GMM model can be employed to estimate the PDF of some samples (like a parametric density estimator). Here, you should train an individual GMM model (with K Components) for each class. Therefore, N GMM models will be created where N shows the number of classes. The label of a sample can be determined using Maximum Likelihood (ML) criteria. In another words, you should find the likelihood of a sample in all classes and then select the class with the maximum likelihood as the label of the sample.


Datasets: vehicle (xaa.dat), Heart (heart.dat)
