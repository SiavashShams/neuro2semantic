import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import vec2text
from collections import defaultdict
import openai
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-goes-here"


class NeuralDataset(Dataset):
    def __init__(
        self,
        trials,
        info,
        sig_elecs,
        subjects,
        bands=['delta'],
        level='word',
        padding=0,
        n=1,
        embedding_model_name='gtr-t5-base',
        stride=1
    ):
        """
        Initializes the NeuralDataset.

        Args:
            trials (list): List of trial data.
            info (dict): Information dictionary containing word mappings.
            sig_elecs (np.ndarray): Significant electrodes mask.
            subjects (list or str): List of subject identifiers.
            bands (list, optional): Frequency bands to consider. Defaults to ['delta'].
            level (str, optional): Processing level ('word', 'sentence', 'custom'). Defaults to 'word'.
            padding (int, optional): Padding size for segments. Defaults to 0.
            n (int, optional): Number of words in n-grams for 'custom' level. Defaults to 1.
            embedding_model_name (str, optional): Name of the embedding model. Defaults to 'gtr-t5-base'.
            stride (int, optional): Stride for n-gram processing. Defaults to 1.
        """
        self.trials = trials
        self.info = info
        self.sig_elecs = sig_elecs
        self.subjects = subjects if isinstance(subjects, list) else [subjects]
        self.bands = bands if isinstance(bands, list) else [bands]
        self.level = level
        self.padding = padding
        self.n = n
        self.stride = stride

        self.embedding_model_name = embedding_model_name
        self.tokenizer, self.encoder = self.load_embedding_model()

        self.neural_data = []
        self.text_embeddings = []
        self.original_texts = []
        self.input_dim = None
        self.embedding_dim = 768  # To be dynamically set based on the model
        self.prepare_data()

    def load_embedding_model(self):
        """
        Loads the specified embedding model.

        Returns:
            tuple: Tokenizer and encoder models or (None, None) for OpenAI embeddings.
        """
        if self.embedding_model_name == "gtr-t5-base":
            encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
            return tokenizer, encoder
        elif self.embedding_model_name == "text-embedding-ada-002":
            return None, None  # OpenAI API handles embeddings
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_name}")

    def prepare_data(self):
        """
        Prepares the dataset by processing each trial based on the specified level.
        """
        for trial in self.trials:
            script = trial['script']
            wrd_labels = trial['wrd_labels']
            subject_electrodes = np.array(trial['subj'])

            significant_mask = self.sig_elecs
            subject_mask = np.isin(subject_electrodes, self.subjects) & significant_mask
            neural_signals = []
            for band in self.bands:
                neural_signals.append(trial[band][:, subject_mask])

            combined_signal = np.concatenate(neural_signals, axis=1)

            if self.input_dim is None:
                self.input_dim = combined_signal.shape[1]

            if self.level == 'word':
                self.process_word_level(combined_signal, script, wrd_labels)
            elif self.level == 'sentence':
                self.process_sentence_level(combined_signal, trial['sentence_labels'])
            elif self.level == 'custom':
                self.process_custom_level(combined_signal, script, wrd_labels)

    def process_word_level(self, neural_data, script, wrd_labels):
        """
        Processes data at the word level.

        Args:
            neural_data (np.ndarray): Neural signal data.
            script (str): Script text.
            wrd_labels (np.ndarray): Word labels.
        """
        words = script.split()
        word_dict = self.info['wrd_dict']

        for word_idx, word in enumerate(words):
            word_label = word_dict.get(word)
            if word_label is None or word_label == -1:
                continue

            time_indices = np.where(wrd_labels == word_label)[0]
            if len(time_indices) == 0:
                continue

            segments = np.split(time_indices, np.where(np.diff(time_indices) != 1)[0] + 1)
            for segment in segments:
                if len(segment) == 0:
                    continue

                start_idx = max(0, segment[0] - self.padding)
                end_idx = min(neural_data.shape[0], segment[-1] + self.padding + 1)

                neural_segment = neural_data[start_idx:end_idx, :]
                correct_embedding = self.get_text_embedding(word)

                self.neural_data.append(neural_segment)
                self.text_embeddings.append(correct_embedding)
                self.original_texts.append(word.lower())

    def process_sentence_level(self, neural_data, sentence_labels):
        """
        Processes data at the sentence level.

        Args:
            neural_data (np.ndarray): Neural signal data.
            sentence_labels (list): List of sentence label dictionaries.
        """
        for sentence_info in sentence_labels:
            sentence = sentence_info['sentence']
            indices = sentence_info['indices']

            if len(indices) == 0:
                continue

            start_idx = max(0, indices[0] - self.padding)
            end_idx = min(neural_data.shape[0], indices[-1] + self.padding + 1)

            neural_segment = neural_data[start_idx:end_idx, :]
            correct_embedding = self.get_text_embedding(sentence)

            self.neural_data.append(neural_segment)
            self.text_embeddings.append(correct_embedding)
            self.original_texts.append(sentence)

    def process_custom_level(self, neural_data, script, wrd_labels):
        """
        Processes data at a custom n-gram level.

        Args:
            neural_data (np.ndarray): Neural signal data.
            script (str): Script text.
            wrd_labels (np.ndarray): Word labels.
        """
        words = script.split()
        word_dict = self.info['wrd_dict']

        for word_idx in range(0, len(words) - self.n + 1, self.stride):
            ngram = words[word_idx:word_idx + self.n]
            ngram_labels = [word_dict.get(word) for word in ngram]

            if any(label is None or label == -1 for label in ngram_labels):
                continue

            start_segments = np.split(
                np.where(wrd_labels == ngram_labels[0])[0],
                np.where(np.diff(np.where(wrd_labels == ngram_labels[0])[0]) != 1)[0] + 1
            )

            best_segment = None
            min_distance = float('inf')

            for start_segment in start_segments:
                start_idx = start_segment[0]
                valid_segment = True
                end_idx = start_idx

                for i in range(1, self.n):
                    next_segments = np.split(
                        np.where(wrd_labels == ngram_labels[i])[0],
                        np.where(np.diff(np.where(wrd_labels == ngram_labels[i])[0]) != 1)[0] + 1
                    )

                    valid_next_segments = [seg for seg in next_segments if seg[0] > end_idx]

                    if not valid_next_segments:
                        valid_segment = False
                        break

                    next_segment = valid_next_segments[0]
                    end_idx = next_segment[-1]

                if not valid_segment:
                    continue

                distance = end_idx - start_idx
                if distance < min_distance:
                    min_distance = distance
                    best_segment = (start_idx, end_idx)

            if best_segment is None:
                continue

            start_idx, end_idx = best_segment
            neural_segment = neural_data[
                max(0, start_idx - self.padding):min(neural_data.shape[0], end_idx + self.padding + 1),
                :
            ]

            correct_embedding = self.get_ngram_embedding(ngram)

            self.neural_data.append(neural_segment)
            self.text_embeddings.append(correct_embedding)
            self.original_texts.append(' '.join(ngram).lower())

    def get_text_embedding(self, text):
        """
        Retrieves the embedding for a given text.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Text embedding.
        """
        if self.embedding_model_name == "gtr-t5-base":
            return self.get_gtr_t5_embedding(text)
        elif self.embedding_model_name == "text-embedding-ada-002":
            return self.get_openai_ada_embedding(text)

    def get_gtr_t5_embedding(self, text):
        """
        Obtains embedding using the GTR-T5 model.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Text embedding.
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding="max_length"
        ).to("cuda")
        with torch.no_grad():
            model_output = self.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            hidden_state = model_output.last_hidden_state
            embedding = vec2text.models.model_utils.mean_pool(
                hidden_state, inputs['attention_mask']
            ).squeeze().cpu().numpy()
        return embedding

    def get_openai_ada_embedding(self, text):
        """
        Obtains embedding using OpenAI's Ada model.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Text embedding.
        """
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002",
            encoding_format="float"
        )
        return np.array(response['data'][0]['embedding'])

    def get_ngram_embedding(self, ngram):
        """
        Retrieves embedding for an n-gram.

        Args:
            ngram (list): List of words forming the n-gram.

        Returns:
            np.ndarray: N-gram embedding.
        """
        ngram_text = ' '.join(ngram).lower()
        return self.get_text_embedding(ngram_text)

    def collect_all_ngrams(self):
        """
        Collects all valid n-grams from the trials.

        Returns:
            list: List of n-grams.
        """
        all_ngrams = []
        for trial in self.trials:
            trial_words = trial['script'].split()
            for i in range(0, len(trial_words) - self.n + 1, self.stride):
                ngram = trial_words[i:i + self.n]
                if all(word in self.info['wrd_dict'] for word in ngram):
                    all_ngrams.append(ngram)
        return all_ngrams

    def __len__(self):
        return len(self.neural_data)

    def __getitem__(self, idx):
        neural_segment = torch.tensor(self.neural_data[idx], dtype=torch.float32)
        text_embedding = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
        original_text = self.original_texts[idx]
        return neural_segment, text_embedding, original_text


def pad_collate_fn(batch):
    """
    Collate function to pad neural segments to the same length.

    Args:
        batch (list): List of tuples containing neural segments, text embeddings, and original texts.

    Returns:
        tuple: Padded neural segments, text embeddings, masks, and original texts.
    """
    neural_segments, text_embeddings, original_texts = zip(*batch)
    max_length = max(segment.size(0) for segment in neural_segments)
    
    # Pad neural segments
    neural_segments, text_embeddings, original_texts = zip(*batch)
    max_length = max(segment.size(0) for segment in neural_segments)
    
    # Pad neural segments to the same length
    padded_segments = [torch.nn.functional.pad(segment, (0, 0, 0, max_length - segment.size(0))) for segment in neural_segments]
    masks = [torch.tensor([1] * segment.size(0) + [0] * (max_length - segment.size(0)), dtype=torch.float32) for segment in neural_segments]

    padded_segments = torch.stack(padded_segments)
    text_embeddings = torch.stack(text_embeddings)
    masks = torch.stack(masks)

    return padded_segments, text_embeddings, masks, original_texts

    


def load_data(file_path):
    """
    Loads data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Loaded data.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def create_dataloader(
    file_path,
    subjects,
    level='word',
    batch_size=4,
    bands=['delta'],
    padding=0,
    n=1,
    embedding_model_name='gtr-t5-base',
    leave_out_trials=None
):
    """
    Creates DataLoaders for training and validation.

    Args:
        file_path (str): Path to the data file.
        subjects (list): List of subject identifiers.
        level (str, optional): Processing level. Defaults to 'word'.
        batch_size (int, optional): Batch size. Defaults to 4.
        bands (list, optional): Frequency bands. Defaults to ['delta'].
        padding (int, optional): Padding size. Defaults to 0.
        n (int, optional): Number of words in n-grams. Defaults to 1.
        embedding_model_name (str, optional): Embedding model name. Defaults to 'gtr-t5-base'.
        leave_out_trials (list, optional): Trials to exclude for validation. Defaults to None.

    Returns:
        tuple: Training and validation DataLoaders along with input dimension, or single DataLoader and input dimension.
    """
    g = torch.Generator()
    g.manual_seed(42)
    
    data = load_data(file_path)
    trials = data['trials']
    info = data['info']
    sig_elecs = data["significan_elecs"]
    
    
    if leave_out_trials is not None:
        # Split trials into training and validation sets
        train_trials = [trial for i, trial in enumerate(trials) if i not in leave_out_trials]
        val_trials = [trial for i, trial in enumerate(trials) if i in leave_out_trials]

        # Initialize datasets
        train_dataset = NeuralDataset(
            train_trials, info, sig_elecs, subjects, bands=bands, level=level,
            padding=padding, n=n, embedding_model_name=embedding_model_name
        )
        val_dataset = NeuralDataset(
            val_trials, info, sig_elecs, subjects, bands=bands, level=level,
            padding=padding, n=n, embedding_model_name=embedding_model_name
        )

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=pad_collate_fn, generator=g
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=pad_collate_fn
        )

        return train_dataloader, val_dataloader, train_dataset.input_dim
    else:
        # In case loading the whole data
        dataset = NeuralDataset(
            trials, info, sig_elecs, subjects, bands=bands, level=level,
            padding=padding, n=n, embedding_model_name=embedding_model_name
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=pad_collate_fn, generator=g
        )

        return dataloader, dataset.input_dim


if __name__ == "__main__":
    # Placeholder file path; replace with actual path when using
    file_path = r'/path/to/data.pkl'
    subjects = ['Subject1', 'Subject2']  # Replace with actual subject IDs
    
    # Trials to exclude for validation
    leave_out_trials = [0, 1]  # Example: Exclude first two trials

    train_dataloader, val_dataloader, input_dim = create_dataloader(
        file_path=file_path,
        subjects=subjects,
        level='sentence',
        bands=['delta'],
        padding=0,
        embedding_model_name='text-embedding-ada-002',
        leave_out_trials=leave_out_trials
    )

    # Example loop through the training data
    for i, (neural_data, text_embeddings, masks, labels) in enumerate(train_dataloader):
        print(f"Train Batch {i + 1}")
        print("Neural data shape:", neural_data.shape)
        print("Text embeddings shape:", text_embeddings.shape)
        print("Masks shape:", masks.shape)
        print("Labels:", labels)
        break  # Remove to process more batches

    # Example loop through the validation data
    for i, (neural_data, text_embeddings, masks, labels) in enumerate(val_dataloader):
        print(f"Validation Batch {i + 1}")
        print("Neural data shape:", neural_data.shape)
        print("Text embeddings shape:", text_embeddings.shape)
        print("Masks shape:", masks.shape)
        print("Labels:", labels)
        break  # Remove to process more batches

