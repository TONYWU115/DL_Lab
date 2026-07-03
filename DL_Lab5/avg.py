import re

data = """

"""

seeds = [int(val) for val in re.findall(r"Seed (\d+):", data)]
rewards = [float(val) for val in re.findall(r"reward = ([\d\.]+)", data)]

window_size = 20
max_avg = -1
best_window = (0, 0)

for i in range(len(rewards) - window_size + 1):
    current_window = rewards[i : i + window_size]
    current_avg = sum(current_window) / window_size
    if current_avg > max_avg:
        max_avg = current_avg
        best_window = (i, i + window_size - 1)

print(f"best seeds = ({seeds[best_window[0]]}, {seeds[best_window[1]]})")
print(f"{max_avg=}")