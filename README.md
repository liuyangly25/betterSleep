# Predicting AC Temperature for Improved Sleep Efficiency

This project aims to predict the optimal air conditioning (AC) temperature based on various factors like room temperature, humidity, heart rate, and body movement. The goal is to help senior people sleep better by controlling the temperature in a way that suits their physiological conditions.

## Problem Statement

The task is to predict an AC temperature command based on input features:
- **Room Temperature** (°C)
- **Humidity** (%)
- **Heart Rate** (bpm)
- **Body Movement** (normalized value representing movement)

The model uses these features to predict a continuous value: the ideal AC temperature (°C) that will enhance sleep comfort.

## Dataset

The dataset consists of multiple features:
- **Room Temperature**: The temperature in the room.
- **Humidity**: The humidity level in the room.
- **Heart Rate**: The heart rate of the individual.
- **Body Movement**: The body movement, measured and normalized for the model.

You can replace the example dataset with your own collected data in a similar format.

## Requirements

- Python 3.7+
- TensorFlow (>=2.0)
- scikit-learn
- numpy

To install the required dependencies, run:

```bash
pip install -r requirements.txt
