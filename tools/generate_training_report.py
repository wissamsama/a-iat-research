import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "reports"


def read_text(path):
    return Path(path).read_text(encoding="utf-8-sig")


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def latest_run(train_runs_dir):
    runs = [path for path in train_runs_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not runs:
        raise SystemExit(f"No completed train run found in: {train_runs_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def code_block(title, content):
    return f"## {title}\n\n```text\n{content.strip()}\n```\n"


def min_metric(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return min(values) if values else None


def build_markdown(run_dir):
    run_dir = Path(run_dir)
    summary = read_json(run_dir / "summary.json")
    metrics = read_csv(run_dir / "metrics.csv")
    config_text = read_text(run_dir / "config.yaml")

    config = summary["config"]
    best_epoch = summary.get("best_epoch")
    best_acc = summary.get("best_test_acc")
    min_loss = summary.get("min_test_loss", min_metric(metrics, "test_loss"))
    checkpoint_path = summary.get("best_checkpoint_path")

    code_files = {
        "Code du modele SimpleCNN": PROJECT_DIR / "models" / "simple_cnn.py",
        "Code du registre et checkpointing": PROJECT_DIR / "models" / "registry.py",
        "Code de chargement des modeles": PROJECT_DIR / "models" / "loading.py",
        "Code du dataset et utilitaires": PROJECT_DIR / "training" / "utils.py",
        "Code de gestion des experiences": PROJECT_DIR / "training" / "experiment.py",
        "Code d'entrainement": PROJECT_DIR / "training" / "train.py",
        "Dependances Python": PROJECT_DIR / "requirements.txt",
    }

    lines = [
        "# Rapport complet - Entrainement SimpleCNN CIFAR-10",
        "",
        "## Run analyse",
        "",
        f"- Run id: `{summary['run_id']}`",
        f"- Dossier du run: `{summary['run_dir']}`",
        f"- Checkpoint enrichi: `{checkpoint_path}`",
        f"- Meilleure accuracy test: `{float(best_acc):.4f}` a l'epoch `{best_epoch}`" if best_acc is not None else "- Meilleure accuracy test: `non disponible`",
        f"- Plus petite test loss: `{float(min_loss):.4f}`" if min_loss is not None else "- Plus petite test loss: `non disponible`",
        f"- Temps total: `{float(summary.get('total_time_sec', 0.0)):.2f}` secondes",
        "",
        "## Format du modele sauvegarde",
        "",
        "Le fichier principal du run est maintenant `checkpoint.pth`. Il ne contient pas seulement les poids du modele. Il contient un checkpoint enrichi avec `model_state_dict`, `model_name`, `dataset`, `num_classes`, `class_names`, `input_shape`, `normalization`, `config` et les metriques principales.",
        "",
        "Ce format permet de reconstruire automatiquement le modele avec `models.loading.load_trained_model(...)`, ce qui est plus pratique pour l'inference, les attaques adversariales et les comparaisons experimentales.",
        "",
        "## Dataset et preprocessing",
        "",
        "Le run actuel entraîne `SimpleCNN` sur CIFAR-10. Les images sont RGB 32x32. A l'entrainement, le code applique `RandomHorizontalFlip`, `RandomCrop(32, padding=4)`, `ToTensor` puis `Normalize`. Au test, seules `ToTensor` et `Normalize` sont appliquees.",
        "",
        "La normalisation CIFAR-10 utilise `mean=(0.4914, 0.4822, 0.4465)` et `std=(0.2470, 0.2435, 0.2616)`. Pour les attaques, ce point est critique : l'epsilon et les bornes doivent etre interpretes par rapport a l'echelle d'entree attendue.",
        "",
        "## Architecture, loss et optimisation",
        "",
        "`SimpleCNN` est une baseline convolutionnelle : trois couches convolutionnelles 3x3 avec ReLU et MaxPool, puis deux couches lineaires. La sortie est un vecteur de logits, pas une softmax.",
        "",
        "La loss est `CrossEntropyLoss`, adaptee a la classification multi-classe exclusive. L'optimiseur est `Adam` avec le learning rate de la config. Modifier l'architecture, les augmentations, la normalisation, la loss, l'optimiseur, le learning rate, le batch size, la seed ou le nombre d'epochs peut changer le resultat obtenu.",
        "",
        "## Hyperparametres du run",
        "",
    ]

    for key, value in config.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend([
        "",
        "## Resultats par epoch",
        "",
        "| epoch | train_loss | train_acc | test_loss | test_acc | epoch_time_sec |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in metrics:
        lines.append(
            f"| {row['epoch']} | {float(row['train_loss']):.4f} | {float(row['train_acc']):.4f} | "
            f"{float(row['test_loss']):.4f} | {float(row['test_acc']):.4f} | {float(row['epoch_time_sec']):.3f} |"
        )

    lines.append("")
    lines.append(code_block("Configuration exacte du run", config_text))
    for title, path in code_files.items():
        if path.exists():
            lines.append(code_block(title, read_text(path)))

    return "\n".join(lines)


def markdown_to_pdf(markdown, pdf_path):
    paragraphs = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line:
            paragraphs.append("")
        elif line.startswith("# "):
            paragraphs.append((line[2:], "title"))
        elif line.startswith("## "):
            paragraphs.append((line[3:], "heading"))
        else:
            paragraphs.extend(textwrap.wrap(line, width=100, replace_whitespace=False) or [""])

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.96
        line_height = 0.018
        page_number = 1

        def new_page():
            nonlocal fig, ax, y, page_number
            ax.text(0.5, 0.025, f"Page {page_number}", ha="center", va="bottom", fontsize=8, color="gray")
            pdf.savefig(fig)
            plt.close(fig)
            page_number += 1
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.96

        for item in paragraphs:
            if y < 0.06:
                new_page()
            if item == "":
                y -= line_height * 0.7
                continue
            if isinstance(item, tuple):
                text, kind = item
                size = 16 if kind == "title" else 12
                ax.text(0.06, y, text, ha="left", va="top", fontsize=size, fontweight="bold")
                y -= line_height * (2.0 if kind == "title" else 1.5)
            else:
                family = "monospace" if item.startswith("```") or item.startswith("|") else None
                ax.text(0.06, y, item, ha="left", va="top", fontsize=8.3, family=family)
                y -= line_height

        ax.text(0.5, 0.025, f"Page {page_number}", ha="center", va="bottom", fontsize=8, color="gray")
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate a detailed PDF report for a training run.")
    parser.add_argument("--run-dir", type=Path, help="Run directory to report. Defaults to the latest run in train_runs/.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run(PROJECT_DIR / "train_runs")
    markdown = build_markdown(run_dir)
    run_id = run_dir.name
    md_path = args.output_dir / f"{run_id}_training_report.md"
    pdf_path = args.output_dir / f"{run_id}_training_report.pdf"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    markdown_to_pdf(markdown, pdf_path)
    print(f"Markdown report: {md_path}")
    print(f"PDF report: {pdf_path}")


if __name__ == "__main__":
    main()






