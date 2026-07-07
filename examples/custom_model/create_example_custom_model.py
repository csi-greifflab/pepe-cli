#!/usr/bin/env python3
"""
Example script showing how to create and use a custom model with EmbedAIRR.
"""

import json
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer


class ExampleProteinModel(nn.Module):
    """
    Example protein model for demonstration.
    This is a simple transformer-based model for protein sequences.
    """

    def __init__(
        self, vocab_size=25, hidden_size=768, num_layers=6, num_heads=12, max_length=512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head
        self.output_head = nn.Linear(hidden_size, vocab_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    ):
        """
        Forward pass compatible with EmbedAIRR custom embedder.
        """
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = self.layer_norm(token_embeddings + position_embeddings)

        # Store hidden states if requested
        all_hidden_states = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Apply transformer layers
        for layer in self.transformer.layers:
            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=(
                    ~attention_mask.bool() if attention_mask is not None else None
                ),
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Output logits
        logits = self.output_head(hidden_states)

        # Create output object
        class ModelOutput:
            def __init__(self):
                self.hidden_states = all_hidden_states if output_hidden_states else None
                self.logits = logits
                self.attentions = None  # Not implemented in this example

        return ModelOutput()


def create_example_model_and_tokenizer(
    save_path="./examples/custom_model/example_protein_model",
):
    """
    Create and save an example protein model and tokenizer.
    """
    # Create model
    model = ExampleProteinModel(
        vocab_size=25,
        hidden_size=384,  # Smaller for example
        num_layers=6,
        num_heads=12,
        max_length=512,
    )

    # Initialize weights (in practice, you would train the model)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Save model
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "num_layers": 6,
                "num_attention_heads": 12,
                "hidden_size": 384,
                "vocab_size": 25,
                "max_position_embeddings": 512,
                "model_type": "protein_transformer",
            },
        },
        os.path.join(save_path, "pytorch_model.pt"),
    )

    # Save config
    config = {
        "num_layers": 6,
        "num_attention_heads": 12,
        "hidden_size": 384,
        "vocab_size": 25,
        "max_position_embeddings": 512,
        "model_type": "protein_transformer",
    }

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create tokenizer files
    _create_tokenizer_files(save_path)

    print(f"Example model saved to: {save_path}")
    print(f"Model config: {config}")

    return save_path


def _create_tokenizer_files(save_path):
    """
    Create tokenizer files compatible with AutoTokenizer.from_pretrained().
    """
    # Define protein vocabulary
    vocab = {
        "<pad>": 0,
        "<cls>": 1,
        "<sep>": 2,
        "<unk>": 3,
        "A": 4,
        "R": 5,
        "N": 6,
        "D": 7,
        "C": 8,
        "Q": 9,
        "E": 10,
        "G": 11,
        "H": 12,
        "I": 13,
        "L": 14,
        "K": 15,
        "M": 16,
        "F": 17,
        "P": 18,
        "S": 19,
        "T": 20,
        "W": 21,
        "Y": 22,
        "V": 23,
        "X": 24,  # Unknown amino acid
    }

    # Create tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "BertTokenizer",
        "model_max_length": 512,
        "do_lower_case": False,
        "cls_token": "<cls>",
        "sep_token": "<sep>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_type": "protein_transformer",
    }

    with open(os.path.join(save_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create vocab.json
    with open(os.path.join(save_path, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # Create vocab.txt (for BertTokenizer compatibility)
    vocab_txt_content = []
    for token, id in sorted(vocab.items(), key=lambda x: x[1]):
        vocab_txt_content.append(token)

    with open(os.path.join(save_path, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab_txt_content))

    # Create special_tokens_map.json
    special_tokens_map = {
        "cls_token": "<cls>",
        "sep_token": "<sep>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }

    with open(os.path.join(save_path, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    print(f"✓ Tokenizer files created in {save_path}")
    print("  - tokenizer_config.json")
    print("  - vocab.json")
    print("  - vocab.txt")
    print("  - special_tokens_map.json")


def create_example_fasta(filename="./examples/custom_model/example_sequences.fasta"):
    """
    Create an example FASTA file with protein sequences.
    """
    sequences = [
        (
            "seq1",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGQIA",
        ),
        (
            "seq2",
            "MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG",
        ),
        (
            "seq3",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        ),
        (
            "seq4",
            "MHHHHHHSSGVDLGTENLYFQSMKKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEQHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLHIQEKLLESYIFQGPRDTHYSQKFPQTLKLLDRKQLAGDLSSQFQEAFEQATRQRQAKVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQ",
        ),
        ("seq5", "ARNDCEQGHILKMFPSTWYV"),  # All amino acids
    ]

    with open(filename, "w") as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n{sequence}\n")

    print(f"Example FASTA file created: {filename}")
    return filename


def create_example_substring(filename="./examples/custom_model/example_substring.csv"):
    """
    Create an example substring file.
    """
    substring_data = [
        (
            "seq1",
            "AKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGQIA",
        ),
        (
            "seq2",
            "DGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG",
        ),
        ("seq3", "DKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),
        (
            "seq4",
            "TCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLHIQEKLLESYIFQGPRDTHYSQKFPQTLKLLDRKQLAGDLSSQFQEAFEQATRQRQAKVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQVQWLKRQFRQAGNTDVQRQFQFSQIRLQAGRQAEFQWLQRQFQARAGQIEQTQRQAQ",
        ),
        ("seq5", "ARNDCEQGHILKMFPSTWYV"),
    ]

    with open(filename, "w") as f:
        f.write("sequence_id,substring_aa\n")
        for seq_id, substring in substring_data:
            f.write(f"{seq_id},{substring}\n")

    print(f"Example substring file created: {filename}")
    return filename


def main():
    """
    Main function to create example files and demonstrate usage.
    """
    print("Creating example custom protein model...")

    # Create example model
    model_path = create_example_model_and_tokenizer()

    # Create example data (written to disk for the demo; return paths unused here)
    create_example_fasta()
    create_example_substring()

    # Test loading the model
    print("\nTesting model loading...")
    try:
        model_data = torch.load(
            os.path.join(model_path, "pytorch_model.pt"), map_location="cpu"
        )
        print("✓ Model loaded successfully")
        print(f"✓ Model config: {model_data.get('config', {})}")

        # Test creating the model
        model = ExampleProteinModel(
            vocab_size=25, hidden_size=384, num_layers=6, num_heads=12, max_length=512
        )
        model.load_state_dict(model_data["model"])
        print("✓ Model architecture created and weights loaded")

        # Test tokenizer loading
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(
                "✓ Tokenizer loaded successfully with AutoTokenizer.from_pretrained()"
            )
            print(f"✓ Tokenizer vocab size: {len(tokenizer.get_vocab())}")

            # Test tokenization
            test_sequence = "ARNDCEQGHILKMFPSTWYV"
            tokens = tokenizer(test_sequence, return_tensors="pt")
            print("✓ Tokenization test successful")
            print(f"✓ Input sequence: {test_sequence}")
            print(f"✓ Tokenized shape: {tokens['input_ids'].shape}")

        except Exception as e:
            print(f"✗ Error testing tokenizer: {e}")

        # Test forward pass
        dummy_input = torch.randint(0, 25, (1, 10))
        dummy_mask = torch.ones(1, 10)

        with torch.no_grad():
            output = model(
                dummy_input, attention_mask=dummy_mask, output_hidden_states=True
            )

        print("✓ Forward pass successful")
        print(f"✓ Output logits shape: {output.logits.shape}")
        print(
            f"✓ Number of hidden states: {len(output.hidden_states) if output.hidden_states else 0}"
        )

    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False

    print("\nAll tests passed! The custom model is ready to use with EmbedAIRR.")
    return True


if __name__ == "__main__":
    main()
