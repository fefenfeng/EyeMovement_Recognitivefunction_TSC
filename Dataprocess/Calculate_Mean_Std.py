import statistics


def compute_mean_std(numbers):
    """
    Simple script to calculate mean and standard deviation value of a list.

    Args:
        numbers (list of float): A non-empty list of values.

    Returns:
        tuple of float:
        A tuple containing two float values, mean and std value.
    """
    mean = sum(numbers) / len(numbers)
    std_dev = statistics.stdev(numbers)
    return mean, std_dev

numbers = []

mean, std_dev = compute_mean_std(numbers)
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
