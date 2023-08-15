import statistics

def compute_mean_std(numbers):
    mean = sum(numbers) / len(numbers)
    std_dev = statistics.stdev(numbers)
    return mean, std_dev


numbers = []

mean, std_dev = compute_mean_std(numbers)
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
