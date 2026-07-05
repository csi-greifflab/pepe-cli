import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src"))

from pepe.parse_arguments import parse_arguments, str2bool, str2ints


class TestParseArguments(unittest.TestCase):
    def setUp(self):
        self.argv_backup = sys.argv[:]

    def tearDown(self):
        sys.argv = self.argv_backup

    def _parse(self, extra_args):
        sys.argv = ["pepe"] + extra_args
        return parse_arguments()

    def test_minimal_required_args(self):
        args = self._parse(
            [
                "--model_name",
                "esm2_t6_8M_UR50D",
                "--fasta_path",
                "test.fasta",
                "--output_path",
                "out",
            ]
        )
        self.assertEqual(args.model_name, "esm2_t6_8M_UR50D")
        self.assertEqual(args.fasta_path, "test.fasta")
        self.assertEqual(args.output_path, "out")
        self.assertEqual(args.extract_embeddings, ["mean_pooled"])
        self.assertTrue(args.streaming_output)

    def test_device_defaults_to_cuda(self):
        args = self._parse(
            [
                "--model_name",
                "esm2_t6_8M_UR50D",
                "--fasta_path",
                "test.fasta",
                "--output_path",
                "out",
            ]
        )
        self.assertEqual(args.device, "cuda")

    def test_device_explicit_cpu(self):
        args = self._parse(
            [
                "--model_name",
                "esm2_t6_8M_UR50D",
                "--fasta_path",
                "test.fasta",
                "--output_path",
                "out",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(args.device, "cpu")

    def test_optional_args(self):
        args = self._parse(
            [
                "--experiment_name",
                "exp1",
                "--model_name",
                "esm2_t6_8M_UR50D",
                "--fasta_path",
                "test.fasta",
                "--output_path",
                "out",
                "--extract_embeddings",
                "per_token",
                "mean_pooled",
                "--layers",
                "-1",
                "6",
                "--batch_size",
                "512",
                "--streaming_output",
                "false",
                "--precision",
                "float16",
            ]
        )
        self.assertEqual(args.experiment_name, "exp1")
        self.assertEqual(args.extract_embeddings, ["per_token", "mean_pooled"])
        self.assertEqual(args.layers, [[-1], [6]])
        self.assertEqual(args.batch_size, 512)
        self.assertFalse(args.streaming_output)
        self.assertEqual(args.precision, "float16")


class TestArgumentTypeHelpers(unittest.TestCase):
    def test_str2bool(self):
        self.assertTrue(str2bool("true"))
        self.assertFalse(str2bool("false"))
        self.assertTrue(str2bool(True))

    def test_str2ints(self):
        self.assertIsNone(str2ints("all"))
        self.assertEqual(str2ints("last"), [-1])
        self.assertEqual(str2ints("1 2 3"), [1, 2, 3])
        self.assertEqual(str2ints(5), [5])


if __name__ == "__main__":
    unittest.main()
