import re
import tokenize
import unittest
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMENTED_CODE = re.compile(
    r"^\s*#+\s*(?:"
    r"(?:class|def|if|elif|else|for|while|with)\b.*:|"
    r"(?:return|import|from|assert|raise)\b|"
    r"self\.|"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(|"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*(?:=|\+=|-=|\*=|/=)|"
    r"[\"'][A-Za-z_]\w*[\"']\s*(?::|,)|"
    r"[+*/&|]"
    r")"
)


def full_line_comments(path):
    source = path.read_bytes()
    comments = []
    for token in tokenize.tokenize(BytesIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line = source.decode("utf-8").splitlines()[token.start[0] - 1]
        if line.lstrip().startswith("#"):
            comments.append((token.start[0], token.string))
    return comments


class CommentCleanupTests(unittest.TestCase):
    def test_no_commented_out_code_remains(self):
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            if "tests" in path.parts or "__pycache__" in path.parts:
                continue
            for line_number, comment in full_line_comments(path):
                if COMMENTED_CODE.match(comment):
                    offenders.append(f"{path.relative_to(ROOT)}:{line_number}: {comment}")
        self.assertEqual(offenders, [], "\n" + "\n".join(offenders))

    def test_main_and_active_loss_have_minimal_comments(self):
        limits = {
            ROOT / "main.py": 5,
            ROOT / "loss_sd.py": 4,
        }
        for path, limit in limits.items():
            comments = full_line_comments(path)
            self.assertLessEqual(
                len(comments),
                limit,
                f"{path.name} has {len(comments)} full-line comments; expected <= {limit}",
            )


if __name__ == "__main__":
    unittest.main()
