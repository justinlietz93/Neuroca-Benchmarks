import random, json, pathlib, lorem, hashlib
N = 10_000
out = pathlib.Path("dataset.jsonl").open("w", encoding="utf8")
for i in range(N):
    snippet = lorem.text()
    record = {
        "id": hashlib.md5(snippet.encode()).hexdigest(),
        "content": snippet,
        "metadata": {"file": f"file_{i%100}.py", "line": random.randint(1, 200)},
        "importance": random.random()
    }
    out.write(json.dumps(record) + "\n")
print(f"Wrote {N} records → dataset.jsonl")
