"""python -m tests.test_indexer  (pytest gerekmez, ağ yok)"""

import json

from app.config import Source
from app.indexer import chunk, matches, notebook_text, wanted, web_url


def test_chunk():
    assert chunk("kısa metin", "# r/a.md") == ["# r/a.md\n\nkısa metin"]

    text = "\n\n".join(f"paragraf {i} " + "x" * 80 for i in range(40))
    chunks = chunk(text, "# r/a.md", size=300, overlap=50)
    assert len(chunks) > 5
    for c in chunks:
        assert c.startswith("# r/a.md\n\n")
        assert len(c.split("\n\n", 1)[1]) <= 300
    # örtüşme: her parçanın son kısmı sonrakinde de geçer; hiçbir paragraf başı kaybolmaz
    for a, b in zip(chunks, chunks[1:]):
        assert a.split("\n\n", 1)[1].strip()[-20:] in b
    assert all(f"paragraf {i} " in "".join(chunks) for i in range(40))

    # başlıkta bölünür: ikinci parça "# " ile başlar
    md = "# A\n" + "a" * 250 + "\n# B\n" + "b" * 100
    c = chunk(md, "# r/x.md", size=300, overlap=50)
    assert c[1].split("\n\n", 1)[1].startswith("# B"), c

    # boşluksuz uzun metin de bölünür, sonsuz döngü yok
    assert len(chunk("z" * 1000, "# h", size=300, overlap=50)) >= 4


def test_notebook():
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# Başlık\n", "açıklama"]},
        {"cell_type": "code", "source": "print(1)", "outputs": [{"text": "GİZLİ"}]},
        {"cell_type": "raw", "source": "yok"},
        {"cell_type": "code", "source": ""},
    ]}
    t = notebook_text(json.dumps(nb))
    assert t == "# Başlık\naçıklama\n\n```python\nprint(1)\n```", t


def test_globs():
    inc = ["**/*.md", "**/Dockerfile"]
    assert matches("README.md", inc)
    assert matches("docs/a/b.md", inc)
    assert matches("Dockerfile", inc)
    assert not matches("a.py", inc)
    assert wanted("README.md", inc, [])
    assert not wanted("docs/x.md", inc, ["docs/**"])
    assert not wanted(".git/config.md", inc, [])
    assert not wanted("sub/.git/HEAD.md", inc, [])


def test_web_url():
    gh = Source(name="r", url="https://user:pw@github.com/org/repo.git", branch="dev")
    assert web_url(gh, "a/b.md") == "https://github.com/org/repo/blob/dev/a/b.md"
    az = Source(name="r", url="https://tok@dev.azure.com/o/p/_git/r", provider="azure")
    assert web_url(az, "a/b.md") == "https://dev.azure.com/o/p/_git/r?path=/a/b.md&version=GBmain"
    # on-prem Azure DevOps Server: provider yazılmasa da /_git/ ile anlaşılır
    onprem = Source(name="r", url="https://tfs.sirket.local/tfs/DefaultCollection/MLOps/_git/platform-docs")
    assert web_url(onprem, "docs/grpc.md") == (
        "https://tfs.sirket.local/tfs/DefaultCollection/MLOps/_git/platform-docs?path=/docs/grpc.md&version=GBmain"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
