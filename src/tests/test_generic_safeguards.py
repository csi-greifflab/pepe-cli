import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("src"))

from pepe.embedders.base_embedder import BaseEmbedder
from pepe.embedders.huggingface_embedder import GenericHuggingFaceEmbedder
from pepe.utils import is_character_tokenizer, warn_if_non_character_tokenizer


def _make_args(fasta_path, split_long_sequences=False, extract_embeddings=None):
    class DummyArgs:
        pass

    args = DummyArgs()
    args.fasta_path = fasta_path
    args.output_path = tempfile.mkdtemp()
    args.experiment_name = "test"
    args.model_name = "someuser/tiny-bert"
    args.disable_special_tokens = False
    args.substring_path = None
    args.context = 0
    args.layers = [[-1]]
    args.batch_size = 1024
    args.max_input_length = "max_input_length"
    args.device = "cpu"
    args.extract_embeddings = extract_embeddings or ["mean_pooled"]
    args.discard_padding = False
    args.flatten = False
    args.streaming_output = False
    args.precision = "32"
    args.flush_batches_after = 128
    args.split_long_sequences = split_long_sequences
    args.split_overlap = 50
    args.force_split_length = None
    args.num_workers = 1
    return args


def _mock_model_tokenizer(max_position_embeddings=512, char_level=True):
    config = MagicMock()
    config.model_type = "bert"
    config.max_position_embeddings = max_position_embeddings
    config.num_attention_heads = 8
    config.num_hidden_layers = 2
    config.hidden_size = 64

    model = MagicMock()
    model.config = config
    model.eval = MagicMock()

    probe = "ACDEFGHIKLMNPQRSTVWY"

    def encode(text, add_special_tokens=True):
        if not char_level and text == probe and not add_special_tokens:
            return list(range(5))
        if not add_special_tokens:
            return list(range(len(text)))
        return [0] + list(range(len(text))) + [1]

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {c: i for i, c in enumerate(probe)}
    tokenizer.all_special_ids = [0, 1]
    tokenizer.all_special_tokens = ["[CLS]", "[SEP]"]
    tokenizer.special_tokens_map = {"additional_special_tokens": ["[CLS]"]}
    tokenizer.pad_token = "[PAD]"
    tokenizer.encode = encode
    tokenizer.model_max_length = max_position_embeddings
    return model, tokenizer


class TestGenericLengthSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.fasta_path = os.path.join(cls.test_dir, "long.fasta")
        with open(cls.fasta_path, "w") as f:
            f.write(f">long_prot\n{'A' * 600}\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def _build_embedder(self, split_long_sequences):
        args = _make_args(self.fasta_path, split_long_sequences=split_long_sequences)
        model, tokenizer = _mock_model_tokenizer(max_position_embeddings=512)
        with patch.object(
            GenericHuggingFaceEmbedder,
            "_initialize_model",
            return_value=(model, tokenizer, 8, 2, 64),
        ), patch.object(
            GenericHuggingFaceEmbedder, "_load_data", return_value=(MagicMock(), 512)
        ), patch.object(GenericHuggingFaceEmbedder, "_set_output_objects"):
            return GenericHuggingFaceEmbedder(args)

    def test_splits_when_split_long_sequences_enabled(self):
        embedder = self._build_embedder(split_long_sequences=True)
        self.assertIn("long_prot", embedder.chunks_mapping)
        self.assertGreater(len(embedder.chunks_mapping["long_prot"]), 1)

    def test_warns_when_split_long_sequences_disabled(self):
        with self.assertLogs("src.embedders.base_embedder", level="WARNING") as logs:
            self._build_embedder(split_long_sequences=False)
        self.assertTrue(
            any("exceed the model's maximum allowed length" in msg for msg in logs.output)
        )


class TestGenericAttentionKwargs(unittest.TestCase):
    def test_eager_attn_when_return_contacts(self):
        args = _make_args(
            os.path.join(os.path.dirname(__file__), "test_files", "test.fasta"),
            extract_embeddings=["attention_layer"],
        )
        recorded = []

        mock_model, mock_tokenizer = _mock_model_tokenizer()
        mock_model.to.return_value = mock_model

        def capture_from_pretrained(link, **kwargs):
            recorded.append(dict(kwargs))
            return mock_model

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained = capture_from_pretrained
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with patch(
            "pepe.embedders.huggingface_embedder._import_transformers",
            return_value=(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                mock_auto_model,
                mock_auto_tokenizer,
                MagicMock(),
                MagicMock(),
            ),
        ):
            embedder = GenericHuggingFaceEmbedder.__new__(GenericHuggingFaceEmbedder)
            BaseEmbedder.__init__(embedder, args)
            embedder._initialize_model("someuser/tiny-bert")

        self.assertTrue(recorded)
        self.assertEqual(recorded[0].get("attn_implementation"), "eager")

    def test_no_eager_attn_without_return_contacts(self):
        args = _make_args(
            os.path.join(os.path.dirname(__file__), "test_files", "test.fasta"),
            extract_embeddings=["mean_pooled"],
        )
        recorded = []

        mock_model, mock_tokenizer = _mock_model_tokenizer()
        mock_model.to.return_value = mock_model

        def capture_from_pretrained(link, **kwargs):
            recorded.append(dict(kwargs))
            return mock_model

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained = capture_from_pretrained
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with patch(
            "pepe.embedders.huggingface_embedder._import_transformers",
            return_value=(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                mock_auto_model,
                mock_auto_tokenizer,
                MagicMock(),
                MagicMock(),
            ),
        ):
            embedder = GenericHuggingFaceEmbedder.__new__(GenericHuggingFaceEmbedder)
            BaseEmbedder.__init__(embedder, args)
            embedder._initialize_model("someuser/tiny-bert")

        self.assertTrue(recorded)
        self.assertNotIn("attn_implementation", recorded[0])


class TestCharacterTokenizerDetection(unittest.TestCase):
    def test_char_level_tokenizer_is_detected(self):
        _, tokenizer = _mock_model_tokenizer(char_level=True)
        self.assertTrue(is_character_tokenizer(tokenizer))

    def test_subword_tokenizer_is_detected(self):
        _, tokenizer = _mock_model_tokenizer(char_level=False)
        self.assertFalse(is_character_tokenizer(tokenizer))

    def test_subword_tokenizer_logs_warning(self):
        _, tokenizer = _mock_model_tokenizer(char_level=False)
        with self.assertLogs("src.utils", level="WARNING") as logs:
            warn_if_non_character_tokenizer(tokenizer, "someuser/wordpiece-bert")
        self.assertTrue(any("subword tokenizer" in msg for msg in logs.output))

    def test_char_level_tokenizer_no_warning(self):
        _, tokenizer = _mock_model_tokenizer(char_level=True)
        with patch.object(logging.getLogger("src.utils"), "warning") as mock_warn:
            warn_if_non_character_tokenizer(tokenizer, "someuser/char-bert")
        mock_warn.assert_not_called()


class TestSentinelModelMaxLength(unittest.TestCase):
    def test_sentinel_model_max_length_treated_as_unknown(self):
        embedder = BaseEmbedder.__new__(BaseEmbedder)
        embedder.force_split_length = None
        embedder.model = MagicMock()
        embedder.model.config = MagicMock(spec=[])
        embedder.tokenizer = MagicMock()
        embedder.tokenizer.model_max_length = 1000000000000.0

        with self.assertLogs("src.embedders.base_embedder", level="INFO") as logs:
            result = embedder._get_model_max_allowed()

        self.assertIsNone(result)
        self.assertTrue(any("unknown" in msg.lower() for msg in logs.output))


if __name__ == "__main__":
    unittest.main()
