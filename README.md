# Unsupervised Learning

<br>

## Overview

Unsupervised learning represents a fundamental approach in machine learning and artificial intelligence. Unlike supervised learning, which relies on labeled data, unsupervised learning identifies patterns and structures within data without predefined labels. This repository provides a practical guide to applying unsupervised learning techniques, particularly in the context of Earth Observation (EO) applications.

Unsupervised learning techniques excel in classification tasks where predefined categories are unavailable. By employing these methods, it is possible to uncover patterns, clusters, and structures within datasets. This repository focuses on classification tasks related to Earth Observation data, specifically:

* Discrimination of sea ice and leads using Sentinel-2 optical data.

* Discrimination of sea ice and leads using Sentinel-3 altimetry data.

<br>

## Objectives

* Apply unsupervised learning techniques to classify satellite imagery data.

* Identify patterns and structures within unlabeled Earth Observation data.

* Compare clustering results with official ESA product labels for validation.

* Utilize K-means and Gaussian Mixture Models (GMM) for clustering and classification.

<br>

## Key Features

### K-means Clustering

K-means is a widely used unsupervised learning algorithm for partitioning data into distinct groups. The number of clusters (k) is pre-defined, and the algorithm assigns each data point to the nearest centroid.

#### Advantages of K-means

* **Efficiency**: Computationally efficient, suitable for large datasets.

* **Ease of Interpretation**: Clustering results are straightforward to analyze.

<br>

## Gaussian Mixture Models (GMM)

GMM represents data as a combination of multiple Gaussian distributions, providing probabilistic cluster assignments instead of hard classifications.

### Advantages of GMM

* **Soft Clustering**: Assigns a probability score to each data point for multiple clusters.

* **Flexible Cluster Covariance**: Captures different shapes and structures within the data.

### Application to Earth Observation Data

#### Image Classification

* **Sentinel-2 Optical Data**: Identify sea ice and leads using spectral bands.

* **Sentinel-3 Altimetry Data**: Use waveform features such as peakiness and stack standard deviation (SSD) to classify sea ice and leads.

<br>

## Results

### Comparison with ESA Data

To validate the clustering results, they can be compared against official ESA product labels, where sea ice is labeled as one class and leads as another. The comparison provides insight into the effectiveness of unsupervised learning techniques in correctly identifying surface types. Performance metrics such as accuracy, precision, recall, and F1-score can be used to assess the classification quality.

A well-matched clustering result to ESA data strengthens confidence in the approach, while discrepancies highlight areas for further refinement. This comparison is crucial in determining the reliability of unsupervised methods for real-world applications in Earth Observation.

<br>

### Key Results

* Unsupervised learning successfully identifies patterns in remote sensing data.

* K-means provides fast clustering but may struggle with non-spherical clusters.

* GMM offers greater flexibility in classification but requires more computational power.

* Clustering results align closely with ESA data, validating the effectiveness of these methods.

<br>

## Conclusion

Unsupervised learning provides powerful tools for discovering patterns and structures within large datasets. By applying techniques such as K-means and Gaussian Mixture Models to Earth Observation data, meaningful classifications can be achieved without the need for labeled examples. These methods contribute to the broader goal of automating and enhancing data interpretation in remote sensing and environmental analysis.
