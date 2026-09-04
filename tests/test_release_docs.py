import ast
import re
import struct
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SCRIPTS = (
    "scripts/train_imagenet_lt.sh",
    "scripts/eval_imagenet_lt.sh",
    "scripts/train_inat18.sh",
    "scripts/train_inat18_200ep.sh",
    "scripts/eval_inat18.sh",
)


class ReleaseDocumentationTests(unittest.TestCase):
    def test_release_files_exist(self):
        self.assertTrue(README.is_file())
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertTrue((ROOT / "THIRD_PARTY_NOTICES.md").is_file())
        self.assertTrue((ROOT / "assets" / "framework.png").is_file())

    def test_readme_has_required_sections(self):
        text = README.read_text(encoding="utf-8")
        for heading in (
            "# VICAL: Vicinal Consistency Alignment for Long-Tailed Visual Recognition",
            "## Method",
            "## Main Results",
            "## Installation",
            "## Data Preparation",
            "## Training",
            "## Evaluation",
            "## Citation",
            "## Acknowledgements",
            "## License",
        ):
            self.assertIn(heading, text)

    def test_readme_records_paper_results(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("| ImageNet-LT | ResNeXt-50 | 200 | 72.8 | 60.8 | 42.6 | 62.9 |", text)
        self.assertIn("| iNaturalist 2018 | ResNet-50 | 100 | 74.3 | 77.3 | 76.2 | 76.6 |", text)
        self.assertIn("| iNaturalist 2018 | ResNet-50 | 200 | 75.5 | 78.1 | 77.3 | 77.5 |", text)

    def test_readme_describes_sc_ded_and_ckf_exactly(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn(
            "The mixed-view online prediction is aligned to the average EMA "
            "prediction over the two augmented full-resolution views.",
            text,
        )
        self.assertIn(
            "DED aligns each expert's low-resolution online prediction to the "
            "EMA ensemble consensus formed from two full-resolution views.",
            text,
        )
        self.assertIn("CKF removes only teacher-wrong/student-correct cases.", text)

    def test_readme_identifies_reported_200_epoch_run(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn(
            "resource-constrained configuration actually used for the reported "
            "77.5 result",
            text,
        )
        self.assertIn("global batch size 256 and initial learning rate 0.1", text)
        self.assertIn("global batch size 512 and initial learning rate 0.2", text)

    def test_readme_commands_match_release_scripts(self):
        text = README.read_text(encoding="utf-8")
        for script in SCRIPTS:
            self.assertIn(script, text)
            path = ROOT / script
            self.assertTrue(path.is_file())
            self.assertTrue(path.stat().st_mode & 0o111, f"{script} is not executable")

    def test_training_script_parameters(self):
        expected = {
            "scripts/train_imagenet_lt.sh": ("imagenet", "resnext50", "200", "256", "0.1", "0.0005"),
            "scripts/train_inat18.sh": ("inat", "resnet50", "100", "512", "0.2", "0.0002"),
            "scripts/train_inat18_200ep.sh": ("inat", "resnet50", "200", "256", "0.1", "0.0002"),
        }
        flags = ("--dataset", "--arch", "--epochs", "--batch-size", "--lr", "--wd")
        for script, values in expected.items():
            source = (ROOT / script).read_text(encoding="utf-8").replace("\\\n", " ")
            for flag, value in zip(flags, values):
                self.assertRegex(source, rf"{re.escape(flag)}\s+{re.escape(value)}(?:\s|$)")

    def test_evaluation_script_parameters(self):
        imagenet = (ROOT / "scripts/eval_imagenet_lt.sh").read_text(encoding="utf-8").replace("\\\n", " ")
        for flag, value in (
            ("--dataset", "imagenet"),
            ("--arch", "resnext50"),
            ("--epochs", "200"),
            ("--batch-size", "256"),
            ("--lr", "0.1"),
        ):
            self.assertRegex(imagenet, rf"{re.escape(flag)}\s+{re.escape(value)}(?:\s|$)")

        inat = (ROOT / "scripts/eval_inat18.sh").read_text(encoding="utf-8")
        self.assertRegex(inat, r"100\)\s+BATCH_SIZE=512\s+LEARNING_RATE=0\.2")
        self.assertRegex(inat, r"200\)\s+BATCH_SIZE=256\s+LEARNING_RATE=0\.1")
        for variable in ("EPOCHS", "BATCH_SIZE", "LEARNING_RATE"):
            self.assertIn(f'"${{{variable}}}"', inat)

    def test_requirements_are_exactly_pinned(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            requirements,
            [
                "torch==2.2.2",
                "torchvision==0.17.2",
                "numpy==1.24.3",
                "Pillow==10.4.0",
                "scipy==1.10.1",
            ],
        )

    def test_framework_png_is_readable_and_nontrivial(self):
        data = (ROOT / "assets" / "framework.png").read_bytes()
        self.assertGreater(len(data), 100_000)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")

        offset = 8
        chunks = []
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            chunk_type = data[offset + 4:offset + 8]
            chunk_data = data[offset + 8:offset + 8 + length]
            expected_crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
            self.assertEqual(zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF, expected_crc)
            chunks.append((chunk_type, chunk_data))
            offset += 12 + length

        self.assertEqual(offset, len(data))
        self.assertEqual(chunks[0][0], b"IHDR")
        self.assertEqual(chunks[-1][0], b"IEND")
        width, height = struct.unpack(">II", chunks[0][1][:8])
        self.assertGreater(width, 2_000)
        self.assertGreater(height, 700)

    def test_root_license_is_vical_mit(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("MIT License\n\nCopyright (c) 2026 VICAL Authors"))
        self.assertIn("Permission is hereby granted, free of charge", text)
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"', text)

    def test_required_third_party_notices_are_bundled(self):
        expected = {
            "BALANCED_CONTRASTIVE_LEARNING.txt": ("Copyright (c) 2022 Jg-Zhu", "MIT License"),
            "CLASSIFIER_BALANCING.txt": ("Facebook, Inc. and its affiliates", "BSD License"),
            "OLTR.txt": ("Copyright (c) 2019, Zhongqi Miao", "BSD 3-Clause License"),
            "LDAM_DRW.txt": ("Copyright (c) 2019 Kaidi Cao", "MIT License"),
            "RIDE.txt": ("Copyright (c) 2020 Tony Lian", "MIT License"),
            "TIMM_RANDAUGMENT.txt": ("Copyright 2019 Ross Wightman", "Apache License"),
        }
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for filename, markers in expected.items():
            path = ROOT / "LICENSES" / filename
            self.assertTrue(path.is_file(), f"missing {path}")
            license_text = path.read_text(encoding="utf-8")
            for marker in markers:
                self.assertIn(marker, license_text)
            self.assertIn(f"LICENSES/{filename}", notices)

    def test_source_headers_point_to_bundled_notices(self):
        for relative in (
            "expert_model/fb_resnets/Expert_ResNeXt.py",
            "expert_model/fb_resnets/Expert_ResNet.py",
            "expert_model/fb_resnets/ResNeXt.py",
            "expert_model/fb_resnets/ResNet.py",
        ):
            self.assertIn(
                "LICENSES/CLASSIFIER_BALANCING.txt",
                (ROOT / relative).read_text(encoding="utf-8")[:800],
            )
        self.assertIn(
            "LICENSES/TIMM_RANDAUGMENT.txt",
            (ROOT / "randaugment.py").read_text(encoding="utf-8")[:800],
        )

    def test_evaluation_requires_an_existing_checkpoint_before_setup(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        ast.parse(source)
        empty_guard = "if args.evaluate and not args.resume.strip():"
        missing_guard = "if args.evaluate and not os.path.isfile(args.resume):"
        self.assertIn(empty_guard, source)
        self.assertIn(missing_guard, source)
        self.assertLess(source.index(empty_guard), source.index("args.store_name ="))
        self.assertLess(source.index(missing_guard), source.index("args.store_name ="))
        self.assertIn('parser.error("--evaluate requires a nonempty --resume checkpoint path")', source)
        self.assertIn('parser.error(f"evaluation checkpoint not found: {args.resume}")', source)

    def test_readme_respects_current_release_scope(self):
        text = README.read_text(encoding="utf-8")
        self.assertNotIn("CIFAR-LT", text)
        self.assertNotIn("Google Drive", text)
        self.assertNotIn("Google Docs", text)
        self.assertNotIn("TO" + "DO", text)
        self.assertNotIn("TB" + "D", text)


if __name__ == "__main__":
    unittest.main()
