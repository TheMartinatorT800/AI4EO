# Unsupervised Learning

<br> 

## Introduction

Applying machine learning models to full-sized images presents unique challenges, particularly in segmentation, classification, and object detection. This project provides a structured approach to deploying trained models such as Convolutional Neural Networks, Vision Transformers, and Random Forest classifiers on entire images or specific regions.

By leveraging deep learning and traditional machine learning techniques, this project enables efficient image processing for applications such as environmental monitoring, medical imaging, and remote sensing. The workflow includes preprocessing and data preparation, model loading and inference, full image rollout, and region-based processing.

<br> 

## Introduction to K-means Clustering

K-means clustering is a widely used unsupervised learning algorithm for partitioning data into distinct groups. The number of clusters (k) is pre-defined, and the algorithm assigns each data point to the nearest centroid.

<br> 

**Key Components of K-means**

Choosing K: The number of clusters must be specified before running the algorithm.

Centroid Initialization: The algorithm starts with initial centroids that may affect the final results.

Assignment Step: Each data point is assigned to the nearest centroid.

Update Step: Centroids are adjusted based on the mean of assigned points.

This iterative process continues until the centroids stabilize, ensuring that within-cluster variation is minimized.

<br> 

## Advantages of K-means

**Efficiency**: K-means is computationally efficient, making it suitable for large datasets.

**Ease of Interpretation**: The clustering results are straightforward to analyze and understand.

<br> 

## Introduction to Gaussian Mixture Models (GMM)

Gaussian Mixture Models (GMM) represent data as a combination of multiple Gaussian distributions. Unlike K-means, which assigns each data point to a single cluster, GMM provides probabilistic cluster assignments.

<br> 

## Why Use Gaussian Mixture Models?

**Soft Clustering**: GMM offers a probability distribution for each data point rather than a hard assignment.

**Flexible Cluster Covariance**: Allows for clusters of different sizes and shapes.

<br> 

## Expectation-Maximization (EM) Algorithm in GMM

The EM algorithm refines the parameters of GMM iteratively:

**Expectation Step (E-step)**: Computes the probability of each data point belonging to a cluster.

**Maximization Step (M-step)**: Updates the Gaussian parameters to maximize the likelihood of the data.

This iterative approach continues until the parameters converge to stable values.

<br> 

## Advantages of GMM

**Soft Clustering**: Provides a probabilistic framework for data classification.

**Cluster Shape Flexibility**: Adapts to various cluster structures.



Application to Image Classification

Classification of Sea Ice and Leads

Unsupervised learning techniques can be applied to remote sensing images to classify sea ice and leads. Using data from Sentinel-2 and Sentinel-3, these techniques help identify patterns without labeled training data.

Preprocessing and Data Preparation

To ensure accurate classification, preprocessing steps such as noise reduction, normalization, and feature selection are essential. These steps enhance the reliability of the clustering results.

K-means and GMM in Image Classification

Both K-means and GMM can be applied to extract meaningful clusters from image data. The clustering results can be visualized using classification maps, aiding in the interpretation of the detected patterns.

Altimetry Classification

Sea Ice and Lead Classification Using Altimetry Data

Unsupervised learning methods can also be applied to altimetry data to distinguish between sea ice and leads. Key features, such as peakiness and stack standard deviation (SSD), play a crucial role in classification.

Feature Extraction

Peakiness: Measures the sharpness of waveform peaks, helping to differentiate between surface types.

Stack Standard Deviation (SSD): Represents variations in altimetry signals, aiding in classification.

Clustering Techniques for Altimetry Data

By applying K-means and GMM to extracted features, different surface types can be identified. These clustering results can be compared with official ESA product labels for validation.

Comparison with ESA Data

To validate the clustering results, they can be compared against official ESA product labels, where sea ice is labeled as one class and leads as another. The comparison provides insight into the effectiveness of unsupervised learning techniques in correctly identifying surface types. Performance metrics such as accuracy, precision, recall, and F1-score can be used to assess the classification quality. A well-matched clustering result to ESA data strengthens confidence in the approach, while discrepancies highlight areas for further refinement. This comparison is crucial in determining the reliability of unsupervised methods for real-world applications in Earth Observation.








