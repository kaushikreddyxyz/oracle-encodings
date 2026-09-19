"""Regression checks: task validity and independent cached-query evaluation.

Run: python -m unittest icl_oracle.binding_v2.test_binding_v2
No downloaded model or tokenizer is needed.
"""
import unittest

import numpy as np

from .tasks import make_task


class TaskTests(unittest.TestCase):
    def test_split_coverage_norm_and_repeatability(self):
        for family in ("selection", "parity", "random", "codebook"):
            for shots in (8, 64, 256):
                with self.subTest(family=family, shots=shots):
                    task = make_task(family, 19, shots, 16, include_seen=True)
                    repeat = make_task(family, 19, shots, 16, include_seen=True)
                    self.assertEqual(task["metadata"], repeat["metadata"])
                    for key, value in task.items():
                        if isinstance(value, np.ndarray):
                            np.testing.assert_array_equal(value, repeat[key])
                    self.assertEqual(set(task["demo_y"]), set(range(8)))
                    self.assertEqual(len(task["demo_y"]), shots)
                    for key in ("demo_x", "query_x"):
                        np.testing.assert_allclose(np.linalg.norm(task[key], axis=1), 1, atol=1e-6)
                    seen = task["query_seen"]
                    self.assertFalse(np.intersect1d(task["demo_codes"], task["query_codes"][~seen]).size)
                    self.assertTrue(np.isin(task["query_codes"][seen], task["demo_codes"]).all())
                    if family != "codebook":
                        self.assertEqual(len(np.unique(task["query_codes"][~seen])), 16)

    def test_labels_match_rule_for_demonstrations_and_queries(self):
        for family in ("selection", "parity"):
            for seed in range(4):
                task = make_task(family, seed, 128, 64)
                rule = task["metadata"]["rule"]
                permutation = np.array(task["metadata"]["label_permutation"])
                for split in ("demo", "query"):
                    bits = task[split + "_bits"]
                    if family == "selection":
                        output = bits[:, rule["coordinates"]]
                    else:
                        pairs = np.array(rule["pairs"])
                        output = bits[:, pairs[:, 0]] ^ bits[:, pairs[:, 1]]
                    expected = permutation[output @ (2 ** np.arange(3))]
                    np.testing.assert_array_equal(task[split + "_y"], expected)

    def test_query_generation_has_no_model_feedback(self):
        # Fixed seed defines both the split and queries before a model runs.
        task = make_task("parity", 23, 64, 32)
        self.assertTrue(np.isin(task["query_codes"], task["metadata"]["heldout_codes"]).all())
        self.assertFalse(np.isin(task["query_codes"], task["metadata"]["demo_pool_codes"]).any())
        np.testing.assert_array_equal(task["query_codes"], make_task("parity", 23, 64, 32)["query_codes"])


class CachedQueryTests(unittest.TestCase):
    def test_matches_full_independent_forwards(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from .runner import Runner

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(41)
            config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                                 num_hidden_layers=3, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=8)
            config._attn_implementation = "eager"
            runner = Runner.__new__(Runner)
            runner.model = Qwen3ForCausalLM(config).eval()
            runner.device, runner.dtype, runner.dim, runner.layer = "cpu", torch.float32, 32, 1
            runner.block = runner.model.model.layers[0]
            task = make_task("selection", 9, 8, 5, include_seen=True)
            prefix, positions = [1], []
            labels = list(range(8, 16))
            for i, y in enumerate(task["demo_y"]):
                positions.append(len(prefix))
                prefix.extend([20 + i, 2, labels[int(y)], 3])
            queries = [[32 + i, 2] for i in range(len(task["query_y"]))]
            layout = dict(prefix=prefix, positions=positions, queries=queries,
                          qpositions=[0] * len(queries), label_ids=labels)
            beta, norm = 0.7, 2.5
            rng = np.random.default_rng(987654 + task["metadata"]["seed"])
            directions = np.linalg.qr(rng.standard_normal((32, 8)))[0].T.astype(np.float32)
            for mode in ("none", "inject"):
                with self.subTest(mode=mode):
                    rows = []
                    runner.evaluate(task, layout, mode, beta, norm, 3, rows.append)
                    self.assertEqual(len(rows), len(queries))
                    # Different query batches must not mutate or append to the shared prefix.
                    for i, query in enumerate(queries):
                        handle = None
                        if mode == "inject":
                            x = np.concatenate([task["demo_x"], task["query_x"][i:i+1]])
                            vectors = (x @ directions) * (beta * norm)
                            handle = runner.hook(positions + [len(prefix)], vectors[None])
                        try:
                            with torch.inference_mode():
                                output = runner.model.model(torch.tensor([prefix + query]), use_cache=False)
                                logits = runner.model.lm_head(output.last_hidden_state[:, -1]).float()[0]
                                expected = torch.softmax(logits[labels], -1).numpy()
                                full = torch.softmax(logits, -1).numpy()
                        finally:
                            if handle is not None:
                                handle.remove()
                        np.testing.assert_allclose(rows[i]["label_probs"], expected, atol=1e-6, rtol=1e-6)
                        self.assertAlmostEqual(rows[i]["label_mass"], float(full[labels].sum()), places=6)
                        self.assertEqual(rows[i]["pred"], int(expected.argmax()))
                        self.assertEqual(rows[i]["correct"], bool(expected.argmax() == task["query_y"][i]))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
