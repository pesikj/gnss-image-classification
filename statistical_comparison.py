import os
import re
from collections import defaultdict

import pandas as pd
from scipy.stats import friedmanchisquare


RESULTS_DIR = "Výsledky"
FILENAME_PATTERN = re.compile(r"resnet(\d{2,3})_\d\.xlsx")


def main() -> None:
    accuracies: dict[str, list[float]] = defaultdict(list)

    for filename in os.listdir(RESULTS_DIR):
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue
        confusion_df = pd.read_excel(os.path.join(RESULTS_DIR, filename))
        confusion_df = confusion_df.iloc[:, 1:]
        cm = confusion_df.values
        accuracy = cm.trace() / cm.sum()
        accuracies[match.group(1)].append(accuracy)

    stat, p_value = friedmanchisquare(*accuracies.values())
    print(f"Friedman statistic: {stat:.3f}, p-value: {p_value:.3f}")

    for model_size, acc_values in sorted(accuracies.items()):
        print(f"resnet{model_size}: {acc_values}")


if __name__ == "__main__":
    main()
