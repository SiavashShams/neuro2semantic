import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import argparse
import wandb
from data_loader import create_dataloader, NeuralDataset
from models import NeuralEncoder
import torch.optim as optim
import random
import numpy as np
import os
from torch.optim.lr_scheduler import OneCycleLR
from reconstruct_text import reconstruct_text, custom_invert_embeddings
from utils import calculate_bert_score, calculate_bleu_score, load_vec2text_model
import vec2text
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from typing import List


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

def calculate_cosine_similarity(neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> float:
    """
    Calculate the average cosine similarity between neural and text embeddings.

    Args:
        neural_embeddings (torch.Tensor): Neural embeddings.
        text_embeddings (torch.Tensor): Text embeddings.

    Returns:
        float: Average cosine similarity.
    """
    return torch.nn.functional.cosine_similarity(neural_embeddings, text_embeddings, dim=-1).mean().item()



def calculate_mse(neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> float:
    """
    Calculate the Mean Squared Error (MSE) between neural and text embeddings.

    Args:
        neural_embeddings (torch.Tensor): Neural embeddings.
        text_embeddings (torch.Tensor): Text embeddings.

    Returns:
        float: MSE value.
    """
    mse = torch.nn.functional.mse_loss(neural_embeddings, text_embeddings)
    return mse.item()

def sequence_loss_fn(reconstructed_texts: List[str], original_texts: List[str], corrector: vec2text.trainers.Corrector) -> torch.Tensor:
    """
    Compute the MSE loss between reconstructed and original texts.

    Args:
        reconstructed_texts (List[str]): List of reconstructed text strings.
        original_texts (List[str]): List of original text strings.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.

    Returns:
        torch.Tensor: MSE loss value.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Tokenize the texts
    tokenized_targets = [
        corrector.tokenizer.encode(text, return_tensors='pt').squeeze(0).to(device) 
        for text in original_texts
    ]
    tokenized_outputs = [
        corrector.tokenizer.encode(recon, return_tensors='pt').squeeze(0).to(device) 
        for recon in reconstructed_texts
    ]

    # Pad sequences to ensure they are the same length
    padded_targets = pad_sequence(tokenized_targets, batch_first=True, padding_value=corrector.tokenizer.pad_token_id)
    padded_outputs = pad_sequence(tokenized_outputs, batch_first=True, padding_value=corrector.tokenizer.pad_token_id)

    # Truncate to the length of the shorter sequence
    min_len = min(padded_outputs.shape[1], padded_targets.shape[1])
    padded_outputs = padded_outputs[:, :min_len].float()  # Convert to float for MSE
    padded_targets = padded_targets[:, :min_len].float()  # Convert to float for MSE

    # Compute MSE loss
    return F.mse_loss(padded_outputs, padded_targets)





class CLIPLoss(nn.Module):
    """
    CLIP Loss module combining cross-entropy losses for neural-to-text and text-to-neural mappings.
    """
    def __init__(self, temperature: float = 0.07):
        """
        Initialize the CLIPLoss module.

        Args:
            temperature (float, optional): Temperature scaling factor. Defaults to 0.07.
        """
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the CLIP loss.

        Args:
            neural_embeddings (torch.Tensor): Neural embeddings.
            text_embeddings (torch.Tensor): Text embeddings.

        Returns:
            torch.Tensor: Computed CLIP loss.
        """
        # Normalize the embeddings
        neural_embeddings = F.normalize(neural_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        logits = (text_embeddings @ neural_embeddings.T) / self.temperature

        # Create target indices
        targets = torch.arange(logits.size(0)).long().to(neural_embeddings.device)

        # Compute cross-entropy loss for both directions
        loss_neural_to_text = F.cross_entropy(logits, targets)
        loss_text_to_neural = F.cross_entropy(logits.T, targets)

        # Average the losses
        loss = (loss_neural_to_text + loss_text_to_neural) / 2.0
        return loss
    
    
class CombinedCLIPTripletLoss(nn.Module):
    """
    Combined loss function that integrates CLIP Loss and Triplet Margin Loss.
    """
    def __init__(self, temperature: float = 0.07, margin: float = 1.0, alpha: float = 0.5):
        """
        Initialize the CombinedCLIPTripletLoss module.

        Args:
            temperature (float, optional): Temperature parameter for CLIP Loss. Defaults to 0.07.
            margin (float, optional): Margin for Triplet Margin Loss. Defaults to 1.0.
            alpha (float, optional): Weighting factor between CLIP Loss and Triplet Margin Loss. Defaults to 0.5.
        """
        super(CombinedCLIPTripletLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the combined CLIP and Triplet loss.

        Args:
            neural_embeddings (torch.Tensor): Neural embeddings.
            text_embeddings (torch.Tensor): Text embeddings.

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Normalize the embeddings
        neural_embeddings = F.normalize(neural_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Compute CLIP Loss
        logits = (text_embeddings @ neural_embeddings.T) / self.temperature
        targets = torch.arange(logits.size(0)).long().to(neural_embeddings.device)
        loss_neural_to_text = self.cross_entropy_loss(logits, targets)
        loss_text_to_neural = self.cross_entropy_loss(logits.T, targets)
        clip_loss = (loss_neural_to_text + loss_text_to_neural) / 2.0

        # Compute Triplet Margin Loss
        anchor_embeddings = neural_embeddings
        positive_embeddings = text_embeddings
        negative_embeddings = text_embeddings[torch.randperm(text_embeddings.size(0))]

        triplet_margin_loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Combine the losses
        combined_loss = self.alpha * clip_loss + (1 - self.alpha) * triplet_margin_loss
        return combined_loss


def cross_entropy_loss(preds: torch.Tensor, targets: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
    """
    Compute the cross-entropy loss.

    Args:
        preds (torch.Tensor): Predictions.
        targets (torch.Tensor): Targets.
        reduction (str, optional): Reduction method. Defaults to 'none'.

    Returns:
        torch.Tensor: Computed loss.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def similarity_loss(reconstructed_texts: List[str], original_texts: List[str], model: SentenceTransformer) -> float:
    """
    Calculate the similarity loss between reconstructed and original texts.

    Args:
        reconstructed_texts (List[str]): List of reconstructed text strings.
        original_texts (List[str]): List of original text strings.
        model (SentenceTransformer): Pre-trained sentence transformer model.

    Returns:
        float: Similarity loss value.
    """
    embeddings_recon = model.encode(reconstructed_texts, convert_to_tensor=True)
    embeddings_orig = model.encode(original_texts, convert_to_tensor=True)
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings_recon, embeddings_orig)
    return 1 - cosine_sim.mean().item()


# This training function is not used during training
def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    corrector: vec2text.trainers.Corrector
) -> tuple:
    """
    Train the neural encoder model for one epoch.

    Args:
        model (nn.Module): Neural encoder model.
        dataloader (DataLoader): Training DataLoader.
        optimizer (optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.

    Returns:
        tuple: Average training loss and average corrector loss.
    """
    model.train()
    corrector.model.train()  # Ensure the corrector is in training mode
    total_loss = 0.0
    total_corrector_loss = 0.0

    for neural_segment, text_embedding, masks, original_texts in dataloader:
        neural_segment = neural_segment.to(device)
        text_embedding = text_embedding.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass through the neural encoder
        neural_features = model(neural_segment, masks)

        # Compute the main loss
        loss = loss_fn(neural_features, text_embedding)

        # Tokenize original texts
        input_ids = corrector.tokenizer(
            list(original_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).input_ids.to(device)

        input_ids = input_ids.long()
        attention_mask = (input_ids != corrector.tokenizer.pad_token_id).long().to(device)

        # Prepare inputs for the corrector
        corrector_inputs = {
            'frozen_embeddings': neural_features.float(),
            'input_ids': input_ids,
            'labels': input_ids.long()
        }

        # Compute corrector loss
        corrector_loss = corrector.compute_loss(corrector.model, corrector_inputs)

        # Combine losses
        combined_loss = loss + corrector_loss
        combined_loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_corrector_loss += corrector_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_corrector_loss = total_corrector_loss / len(dataloader)

    wandb.log({
        "Train Total Loss": avg_loss,
        "Train Corrector Loss": avg_corrector_loss,
        "Epoch": epoch
    })

    print(f"Epoch {epoch}: Avg Train Loss: {avg_loss:.4f}, Avg Corrector Loss: {avg_corrector_loss:.4f}")

    return avg_loss, avg_corrector_loss


def train_neural_encoder(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    embedding_dir: str
) -> float:
    """
    Train the neural encoder model for one epoch.

    Args:
        model (nn.Module): Neural encoder model.
        dataloader (DataLoader): Training DataLoader.
        optimizer (optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        embedding_dir (str): Directory to save embeddings.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0
    all_neural_embeddings = []
    all_text_embeddings = []

    for neural_segment, text_embedding, masks, original_texts in dataloader:
        neural_segment = neural_segment.to(device)
        text_embedding = text_embedding.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass through the neural encoder
        neural_features = model(neural_segment, masks)

        # Compute the main loss
        loss = loss_fn(neural_features, text_embedding)
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()
        all_neural_embeddings.append(neural_features.cpu().detach().numpy())
        all_text_embeddings.append(text_embedding.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)

    wandb.log({
        "Train Total Loss": avg_loss,
        "Epoch": epoch
    })

    print(f"Epoch {epoch}: Avg Train Loss: {avg_loss:.4f}")

    # Save embeddings periodically
    if epoch % 10 == 0:
        np.save(os.path.join(embedding_dir, f"training_neural_embeddings_epoch_{epoch}.npy"), np.concatenate(all_neural_embeddings))
        np.save(os.path.join(embedding_dir, "train_text_embeddings.npy"), np.concatenate(all_text_embeddings))

    return avg_loss


def fine_tune_corrector(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    corrector: vec2text.trainers.Corrector
) -> float:
    """
    Fine-tune the Vec2Text corrector model.

    Args:
        model (nn.Module): Frozen neural encoder model.
        dataloader (DataLoader): Training DataLoader.
        optimizer (optim.Optimizer): Optimizer for the corrector.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.

    Returns:
        float: Average corrector loss.
    """
    model.eval()  # Freeze the neural encoder during corrector fine-tuning
    corrector.model.train()  # Ensure the corrector is in training mode
    total_corrector_loss = 0.0

    for neural_segment, text_embedding, masks, original_texts in dataloader:
        neural_segment = neural_segment.to(device)
        text_embedding = text_embedding.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            # Forward pass through the frozen neural encoder
            neural_features = model(neural_segment, masks)

        # Tokenize original texts
        input_ids = corrector.tokenizer(
            list(original_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).input_ids.to(device)

        input_ids = input_ids.long()
        attention_mask = (input_ids != corrector.tokenizer.pad_token_id).long().to(device)

        # Prepare inputs for the corrector
        corrector_inputs = {
            'frozen_embeddings': neural_features.float(),
            'input_ids': input_ids,
            'labels': input_ids.long()
        }

        # Compute corrector loss
        corrector_loss = corrector.compute_loss(corrector.model, corrector_inputs)

        # Backpropagate and update corrector parameters
        corrector_loss.backward()
        optimizer.step()

        total_corrector_loss += corrector_loss.item()

    avg_corrector_loss = total_corrector_loss / len(dataloader)

    wandb.log({
        "Train Corrector Loss": avg_corrector_loss,
        "Epoch": epoch
    })

    print(f"Epoch {epoch}: Avg Corrector Loss: {avg_corrector_loss:.4f}")

    return avg_corrector_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    corrector: vec2text.trainers.Corrector,
    similarity_model: SentenceTransformer,
    n_steps: int,
    save_embeddings: bool = False,
    save_freq: int = 10,
    output_dir: str = "embeddings"
) -> tuple:
    """
    Evaluate the neural encoder model on the validation set.

    Args:
        model (nn.Module): Neural encoder model.
        dataloader (DataLoader): Validation DataLoader.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
        epoch (int): Current epoch number.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.
        similarity_model (SentenceTransformer): Sentence transformer for similarity calculations.
        n_steps (int): Number of steps for reconstruction.
        save_embeddings (bool, optional): Whether to save embeddings. Defaults to False.
        save_freq (int, optional): Frequency to save embeddings. Defaults to 10.
        output_dir (str, optional): Directory to save embeddings. Defaults to "embeddings".

    Returns:
        tuple: Average loss, average cosine similarity, neural embeddings, text embeddings, average BERT score.
    """
    model.eval()
    corrector.model.eval()
    total_loss = 0.0
    total_similarity_loss = 0.0
    total_cosine_similarity = 0.0
    total_mse_loss = 0.0
    total_corrector_loss = 0.0
    total_bleu_score = 0.0
    total_bert_score = 0.0
    max_bert_score = -float('inf')
    best_reconstructed_text = None
    best_original_text = None

    all_neural_embeddings = []
    all_text_embeddings = []

    with torch.no_grad():
        for neural_segment, text_embedding, masks, original_texts in dataloader:
            neural_segment = neural_segment.to(device)
            text_embedding = text_embedding.to(device)
            masks = masks.to(device)

            # Forward pass through the neural encoder
            neural_features = model(neural_segment, masks)

            # Compute the main loss
            loss = loss_fn(neural_features, text_embedding)
            total_loss += loss.item()

            # Tokenize original texts
            input_ids = corrector.tokenizer(
                list(original_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).input_ids.to(device)

            input_ids = input_ids.long()
            attention_mask = (input_ids != corrector.tokenizer.pad_token_id).long().to(device)

            # Prepare inputs for the corrector
            corrector_inputs = {
                'frozen_embeddings': neural_features.float(),
                'input_ids': input_ids,
                'labels': input_ids.long()
            }

            # Compute corrector loss
            corrector_loss = corrector.compute_loss(corrector.model, corrector_inputs)
            total_corrector_loss += corrector_loss.item()

            # Reconstruct the sentence using the corrector
            reconstructed_texts = reconstruct_text(neural_features, corrector, n_steps=n_steps)

            # Calculate similarity loss
            sim_loss = similarity_loss(reconstructed_texts, original_texts, similarity_model)
            total_similarity_loss += sim_loss

            # Compute additional metrics
            cosine_similarity = calculate_cosine_similarity(neural_features, text_embedding)
            mse = calculate_mse(neural_features, text_embedding)

            total_cosine_similarity += cosine_similarity
            total_mse_loss += mse

            bleu_score = calculate_bleu_score(reconstructed_texts, original_texts)
            bert_scores = calculate_bert_score(reconstructed_texts, original_texts)
            total_bleu_score += bleu_score
            total_bert_score += bert_scores[2]  

            # Track the best BERT score
            if bert_scores[2] > max_bert_score:
                max_bert_score = bert_scores[2]
                best_reconstructed_text = reconstructed_texts
                best_original_text = original_texts

            # Store embeddings if needed
            if save_embeddings:
                all_neural_embeddings.append(neural_features.cpu().numpy())
                all_text_embeddings.append(text_embedding.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_similarity_loss = total_similarity_loss / len(dataloader)
    avg_cosine_similarity = total_cosine_similarity / len(dataloader)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_bleu_score = total_bleu_score / len(dataloader)
    avg_bert_score = total_bert_score / len(dataloader)

    print(f"Epoch {epoch}: Avg Eval Loss: {avg_loss:.4f}, Avg Similarity Loss: {avg_similarity_loss:.4f}, "
          f"Cosine Similarity: {avg_cosine_similarity:.4f}, BLEU: {avg_bleu_score:.4f}, BERT: {avg_bert_score:.4f}")

    if best_reconstructed_text and best_original_text:
        print(f"Best Reconstructed Text with BERT Score {max_bert_score:.4f}: {best_reconstructed_text}")
        print(f"Original Text: {best_original_text}")

    wandb.log({
        "Eval Loss": avg_loss,
        "Eval Corrector Loss": total_corrector_loss / len(dataloader),
        "Eval Similarity Loss": avg_similarity_loss,
        "Eval Cosine Similarity": avg_cosine_similarity,
        "Eval MSE": avg_mse_loss,
        "Eval BLEU Score": avg_bleu_score,
        "Eval BERT Score": avg_bert_score,
        "Epoch": epoch
    })

    return avg_loss, avg_cosine_similarity, all_neural_embeddings, all_text_embeddings, avg_bert_score



def main(args):
    """
    Main function to train the neural encoder model with integrated Vec2Text corrector.

    Args:
        args: Parsed command-line arguments.
    """
    # Initialize WandB
    set_seed(42)
    print("Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, settings=wandb.Settings(_service_wait=60))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
    
    # Load data
    if args.leave_out_trials is not None:
        train_loader, val_loader, input_dim = create_dataloader(
            args.file_path,
            subjects=args.subjects,
            level=args.level,
            bands=args.bands,
            padding=args.padding,
            n=args.ngram,
            embedding_model_name=args.embedding_model_name,
            leave_out_trials=args.leave_out_trials
        )
    else:
        dataloader, input_dim = create_dataloader(
            args.file_path,
            subjects=args.subjects,
            level=args.level,
            bands=args.bands,
            padding=args.padding,
            n=args.ngram,
            embedding_model_name=args.embedding_model_name
        )

        # Split data into training and validation sets
        dataset_size = len(dataloader.dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(dataloader.dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=dataloader.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=dataloader.collate_fn
        )
    
    # Initialize model, optimizer, and loss function
    if args.embedding_model_name == "gtr-t5-base":
        embedding_dim = 768
        vec2text_model = 'gtr-base'
    else:
        embedding_dim = 1536
        vec2text_model = 'text-embedding-ada-002'

    model = NeuralEncoder(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    loss_fn = CombinedCLIPTripletLoss(temperature=args.temperature, margin=args.margin, alpha=args.alpha)

    # Load Vec2Text corrector
    corrector = load_vec2text_model(vec2text_model)

    # Ensure all corrector model parameters are trainable
    for param in corrector.model.parameters():
        param.requires_grad = True

    # Initialize optimizers
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Set up directories for saving results
    experiment_dir = os.path.join("results", args.wandb_run_name)
    embedding_dir = os.path.join(experiment_dir, "embeddings")
    model_dir = os.path.join(experiment_dir, "model")
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    best_eval_loss = float('inf')
    best_cosine_similarity = float('-inf')

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        # Train the neural encoder
        train_loss = train_neural_encoder(
            model, train_loader, model_optimizer, loss_fn, device, epoch, embedding_dir
        )

        # Evaluate the model
        val_loss, val_cosine_similarity, all_neural_embeddings, all_text_embeddings, avg_bert_score = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            epoch,
            corrector,
            similarity_model,
            n_steps=args.n_steps,
            save_embeddings=True,
            save_freq=10,
            output_dir=embedding_dir
        )

        # Save model based on evaluation metrics
        if val_loss < best_eval_loss:
            best_eval_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_eval_loss_model.pth"))
            np.save(os.path.join(embedding_dir, "best_eval_loss_neural_embeddings.npy"), np.concatenate(all_neural_embeddings))
            np.save(os.path.join(embedding_dir, "text_embeddings.npy"), np.concatenate(all_text_embeddings))
            print(f"Saved model and embeddings with best evaluation loss at epoch {epoch}")

        if val_cosine_similarity > best_cosine_similarity:
            best_cosine_similarity = val_cosine_similarity
            torch.save(model.state_dict(), os.path.join(model_dir, "best_cosine_similarity_model.pth"))
            np.save(os.path.join(embedding_dir, "best_cosine_similarity_neural_embeddings.npy"), np.concatenate(all_neural_embeddings))
            print(f"Saved model and embeddings with best cosine similarity at epoch {epoch}")


    # Save the final model checkpoint
    torch.save(model.state_dict(), os.path.join(model_dir, "neural_encoder_final.pth"))

    # Additional fine-tuning for the corrector
    optimizer = torch.optim.AdamW(corrector.model.parameters(), lr=args.corrector_lr)
    best_bert_score = float('-inf')

    for epoch in range(1, 3):
        fine_tune_corrector(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            corrector
        )
        val_loss, val_cosine_similarity, all_neural_embeddings, all_text_embeddings, avg_bert_score = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            epoch,
            corrector,
            similarity_model,
            n_steps=args.n_steps,
            save_embeddings=True,
            save_freq=10,
            output_dir=embedding_dir
        )

        if avg_bert_score > best_bert_score:
            best_bert_score = avg_bert_score
            torch.save(corrector.model.state_dict(), os.path.join(model_dir, "best_corrector_bert_score_model.pth"))
            np.save(os.path.join(embedding_dir, "best_bert_score_neural_embeddings.npy"), np.concatenate(all_neural_embeddings))
            np.save(os.path.join(embedding_dir, "best_bert_score_text_embeddings.npy"), np.concatenate(all_text_embeddings))
            print(f"Saved corrector model and embeddings with best BERT score at epoch {epoch}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Encoder with CLIP Loss")
    parser.add_argument(
        '--file_path',
        type=str,
        default='/path/to/data.pkl',
        help='Path to the data file'
    )
    parser.add_argument(
        "--subjects",
        nargs='+',
        default=["Subject1 Subject2 Subject3"],
        help="List of subjects"
    )
    parser.add_argument(
        "--bands",
        nargs='+',
        default=["highgamma"],
        help="List of frequency bands"
    )
    parser.add_argument(
        "--level",
        type=str,
        default="sentence",
        help="Level of processing: word, sentence, or custom"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding around segments"
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=10,
        help="N-gram size for custom level"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="Number of steps for reconstruction"
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="gtr-t5-base",
        help="Embedding model to use"
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default="neuro2semantic",
        help='WandB project name'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default="default_run",
        help='WandB run name'
    )
    parser.add_argument(
        "--leave_out_trials",
        nargs='+',
        type=int,
        default=[0],
        help="Indices of trials to leave out for evaluation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0013,
        help="Learning rate for the neural encoder"
    )
    parser.add_argument(
        "--corrector_lr",
        type=float,
        default=0.0013,
        help="Learning rate for the corrector model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter for CLIP loss"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="Frequency (in epochs) to save model checkpoints"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for n-gram processing"
    )
    parser.add_argument(
        '--vec2text_model_name',
        type=str,
        default="gtr-base",
        help="Vec2Text model to use for reconstruction"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help="Margin for Triplet Margin Loss"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for combining CLIP and Triplet losses"
    )

    args = parser.parse_args()

    main(args)
    
    