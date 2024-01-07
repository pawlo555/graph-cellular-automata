import subprocess

for n in [10, 20, 30, 40, 50, 60]:
    for e in [20, 40, 60, 80, 100]:
        print(f"nodes: {n}, edges: {e}")
        subprocess.run(f"python test.py --nodes {n} --edges {e}")
