import csv

def save_to_csv(results, csv_path):
    """
    Save results of analysis to a CSV file (-save tag) 
    """
    if not results:
        return

    keys = results[0].keys()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results have been saved to {csv_path}")
