## Disclaimer

This dataset consists of synthetic neural recordings generated for the purpose of demonstrating data structure and format. Due to patient privacy concerns, we are unable to share real iEEG neural data. The neural signals provided in this dataset are randomly generated normal variables and are not representative of actual neural activity. This dataset is solely intended to illustrate how our real data is organized and formatted.

## Overview

This dataset consists of synthetic neural recordings and corresponding text labels, intended for experiments involving neural signal processing and text embedding models. The dataset is structured in three main components: `trials`, `info`, and `significan_elecs`. Each component contains critical information for training and evaluating models that map neural signals to text embeddings.

## Data Structure

### 1. `trials` (list of dictionaries)

Each entry in the `trials` list represents a single trial of neural data recording associated with a particular text input. Below is an interpretation of the keys found in each trial dictionary:

- **script**: A string containing the text or script associated with the neural data for this trial. This script is what the subject have heard, and it is used to generate word embeddings.
  
  - Example: `"The cat sat on the mat"`

- **wrd_labels**: A NumPy array of integers, where each value represents a unique label for a word in the script. These labels link words in the `script` to their respective neural data points. The labels can be used to align text with neural activity in time.
  
  - Example: `[12, 13, 14, 15]`

- **delta, highgamma, [other bands]**: NumPy arrays where each row represents the neural signal over time, and each column corresponds to a neural electrode. Each band represents neural activity filtered within a specific frequency range (e.g., `delta`, `highgamma`). These signals are aligned with the time points where specific words or sentences are presented in the script.
  
  - Example shape: `(time_points, electrodes)`

- **phn_labels**: A NumPy array representing the phoneme labels corresponding to the script. These labels can be used to analyze neural responses at the phonetic level.
  
  - Example: `[4, 3, 2, 1]`

- **subject**: A list of strings containing the subject IDs for the data in this trial. This field can be used to identify which electrodes correspond to which subjects.
  
  - Example: `["Subject1"]`

### 2. `info` (dictionary)

The `info` dictionary contains metadata and mappings used throughout the dataset. Key entries include:

- **wrd_dict**: A dictionary mapping words to unique integer labels. This mapping is used to create the `wrd_labels` in the `trials`.
  
  - Example: `{'cat': 12, 'mat': 13}`

- **phn_label_list**: A list of all phoneme labels used in the dataset. Each phoneme in the `phn_labels` arrays in the `trials` corresponds to one of these labels.
  
  - Example: `['AA', 'AE', 'AH', 'AO']`

- **manner_label_list**: A list of articulation manner labels corresponding to the `manner_labels` in the `trials`.

- **wrd_labels**: A list or array mapping word labels to specific words, the reverse of `wrd_dict`.

### 3. `significan_elecs` (NumPy array)

The `significan_elecs` array is a binary mask where each entry indicates whether a particular electrode is significant for the trial (True for significant, False for not significant). This can be used for selecting relevant neural data during processing.
