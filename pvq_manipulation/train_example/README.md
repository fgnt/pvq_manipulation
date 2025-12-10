# Example Training

This example demonstrates how to train a Conditional Normalizing Flow using a toy dataset. The training data is based on the "moons" dataset, which consists of two classes. The flow learns a conditional mapping from the target distribution to the base distribution (a standard normal distribution) using the class label as a condition.

## Dataset
The trainer requires an iterable dataset. In this toy example, the dataset is generated as follows:
1. The observations are stored in separate files.
2. A JSON file is created to list all the files and their corresponding metadata.
3. The JSON file is loaded into a dictionary with the following structure:

```sh
{
  "train": {
    "example_id_0": {
      "label": [float, float, ...],  // List of float values representing the condition
      "observation": file_path_to_observation  // Path to the file containing the observation data
    },
    "example_id_1": {
      "label": [float, float, ...],
      "observation": file_path_to_observation  
    },
    "example_id_2": {
      "label": [float, float, ...],
      "observation": file_path_to_observation  
    },
    ...
  },
  "eval": {
    "example_id_0": {
      "label": [float, float, ...],
      "observation": file_path_to_observation  
    },
    ...
  },
  "test": {
    "example_id_0": {
      "label": [float, float, ...],
      "observation": file_path_to_observation 
    },
    ...
  }
}
```
From the dictionary, a lazy dataset is created, and the observation files are loaded on-the-fly during the pipeline. To adapt the training for speaker embeddings and voice qualities, the observation field should contain the preloaded speaker embeddings, and the label field should represent the estimated voice quality strength (normalized between 0 and 1). Additionally, update the configuration to use config_flow_speech_manipulation.yaml. Example code for extracting these two features can be found in Example_Notebook.ipynb. 
normalizing_flow.ipynb demonstrates how to load the trained model and how to apply the normalizing flow.