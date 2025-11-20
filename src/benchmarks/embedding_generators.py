#!/usr/bin/env python
"""
Embedding generators for TMvec benchmarking.
"""

from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class EmbeddingGenerator(ABC):
    """Base class for protein embedding generators."""

    @abstractmethod
    def generate(self, sequences, batch_size=32, max_length=512, device='cuda'):
        """Generate embeddings for protein sequences."""
        pass

class ProtT5EmbeddingGenerator(EmbeddingGenerator):
    """Generate ProtT5 embeddings for protein sequences."""

    def generate(self, sequences, batch_size=32, max_length=512, device='cuda'):
        from transformers import T5Tokenizer, T5EncoderModel

        print("Generating ProtT5 embeddings...")
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        model.to(device)
        model.eval()

        all_embeddings = []

        sequences_spaced = [" ".join(list(seq)) for seq in sequences]

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences_spaced), batch_size)):
                batch_seqs = sequences_spaced[i:i + batch_size]

                encoded = tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                embeddings = outputs.last_hidden_state

                all_embeddings.append(embeddings.cpu())

        print(f"Generated ProtT5 embeddings: {all_embeddings[0].shape}")
        return all_embeddings
