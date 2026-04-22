import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    for file_path in os.listdir("."):
        if not re.match(r"test_statistics_\d+\.csv", file_path):
            continue

        data = pd.read_csv(file_path)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        sns.lineplot(
            data=data,
            x="epoch", y="epoch_loss", hue="phase",
            style="cross_validation_iteration",
            markers=True, dashes=False, ax=axes[0],
        )
        axes[0].set_title("Cross-Validation Loss per Epoch")
        axes[0].set_ylabel("Loss (-)")

        legend = axes[0].legend(title=None)
        for legend_text in legend.get_texts():
            if "cross_validation_iteration" in legend_text.get_text():
                legend_text.set_text(
                    legend_text.get_text().replace("cross_validation_iteration", "subset")
                )

        sns.lineplot(
            data=data,
            x="epoch", y="epoch_acc", hue="phase",
            style="cross_validation_iteration",
            markers=True, dashes=False, ax=axes[1], legend=False,
        )
        axes[1].set_title("Cross-Validation Accuracy per Epoch")
        axes[1].set_ylabel("Accuracy (-)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0.7, 1)

        plt.tight_layout()

        pdf_file_path = file_path.replace(".csv", ".pdf")
        fig.savefig(pdf_file_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
