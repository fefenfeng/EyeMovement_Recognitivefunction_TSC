import statistics

def compute_mean_std(numbers):
    mean = sum(numbers) / len(numbers)
    std_dev = statistics.stdev(numbers)
    return mean, std_dev


numbers = [0.8667, 1, 1, 0.9655, 1,
           0.9, 0.9333, 0.9655, 0.8966, 0.931,
           0.8667, 0.9667, 0.931, 0.8966, 1]

mean, std_dev = compute_mean_std(numbers)
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
