import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime

sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def load_json_files(directory, pattern="*.json"):
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    data = []
    for f in files:
        try:
            with open(f) as fp:
                data.append(json.load(fp))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return data

def plot_anomaly_timeline(anomaly_dir, fault_dir, outdir):
    anomalies = load_json_files(anomaly_dir, "all_anomalies_*.json")
    faults = load_json_files(fault_dir, "*.json")
    anomaly_times = []
    for batch in anomalies:
        for a in batch:
            anomaly_times.append(a.get("timestamp", 0))
    fault_times = [f.get("timestamp", 0) for f in faults]
    print(f"[Anomaly Timeline] Found {len(anomaly_times)} anomalies and {len(fault_times)} faults.")
    if not anomaly_times and not fault_times:
        print("No anomaly or fault data to plot, skipping.")
        return
    plt.figure(figsize=(10, 3))
    plt.hist(anomaly_times, bins=50, alpha=0.7, label="Anomalies")
    plt.vlines(fault_times, ymin=0, ymax=plt.ylim()[1] if plt.ylim()[1] > 0 else 1, color='r', linestyle='dashed', label="Faults")
    plt.xlabel("Time (UNIX timestamp)")
    plt.ylabel("Count")
    plt.title("Anomaly and Fault Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "anomaly_fault_timeline.png"))
    plt.close()

def plot_graph_evolution(graph_dir, outdir):
    files = sorted(glob.glob(os.path.join(graph_dir, "graph_*.json")))
    times, nodes, edges = [], [], []
    for f in files:
        with open(f) as fp:
            g = json.load(fp)
            times.append(g.get("timestamp", 0))
            nodes.append(len(g.get("nodes", [])))
            edges.append(len(g.get("edges", [])))
    print(f"[Graph Evolution] Found {len(times)} graph snapshots.")
    if not times:
        print("No graph snapshots to plot, skipping.")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(times, nodes, label="Nodes")
    plt.plot(times, edges, label="Edges")
    plt.xlabel("Time (UNIX timestamp)")
    plt.ylabel("Count")
    plt.title("Graph Evolution Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "graph_evolution.png"))
    plt.close()

def plot_evaluation_metrics(eval_dir, outdir):
    files = glob.glob(os.path.join(eval_dir, "*.json"))
    metrics = []
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
                d["file"] = os.path.basename(f)
                metrics.append(d)
        except Exception:
            continue
    print(f"[Evaluation Metrics] Found {len(metrics)} evaluation files.")
    if not metrics:
        print("No evaluation metrics to plot, skipping.")
        return
    df = pd.DataFrame(metrics)
    for col in df.columns:
        if col not in ["file"] and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notnull().sum() == 0:
                print(f"No data for metric {col}, skipping plot.")
                continue
            print(f"  Plotting metric: {col} (non-null: {df[col].notnull().sum()})")
            plt.figure(figsize=(7, 3))
            sns.barplot(x="file", y=col, data=df)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Evaluation Metric: {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"eval_{col}.png"))
            plt.close()

def plot_service_health(monitor_dir, outdir):
    files = sorted(glob.glob(os.path.join(monitor_dir, "health_snapshot_*.json")))
    times, healthy, unhealthy, unknown = [], [], [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
            # Extract timestamp from the first service entry
            first_service = next(iter(d.values()), None)
            if first_service and "timestamp" in first_service:
                # Convert ISO timestamp to UNIX timestamp
                t = datetime.fromisoformat(first_service["timestamp"]).timestamp()
            else:
                t = 0
            h, u, unk = 0, 0, 0
            for s in d.values():
                status = s.get("status", s.get("health", "unknown"))
                if status == "healthy":
                    h += 1
                elif status == "unhealthy":
                    u += 1
                else:
                    unk += 1
            times.append(t)
            healthy.append(h)
            unhealthy.append(u)
            unknown.append(unk)
    print(f"[Service Health] Found {len(times)} monitoring snapshots.")
    if not times:
        print("No service health data to plot, skipping.")
        return
    plt.figure(figsize=(10, 4))
    plt.plot(times, healthy, label="Healthy")
    plt.plot(times, unhealthy, label="Unhealthy")
    plt.plot(times, unknown, label="Unknown")
    plt.xlabel("Time (UNIX timestamp)")
    plt.ylabel("Service Count")
    plt.title("Service Health Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "service_health_timeline.png"))
    plt.close()

def plot_fault_injection_events(fault_injection_dir, outdir):
    events_file = os.path.join(fault_injection_dir, "fault_events.jsonl")
    if not os.path.exists(events_file):
        print("[Fault Injection Events] No fault_events.jsonl file found.")
        return
    times, types = [], []
    with open(events_file) as fp:
        for line in fp:
            try:
                d = json.loads(line)
                times.append(d.get("timestamp", 0))
                # Try to get type from top-level, else from fault_info
                fault_type = d.get("type")
                if not fault_type and "fault_info" in d:
                    fault_type = d["fault_info"].get("type", "unknown")
                types.append(fault_type if fault_type else "unknown")
            except Exception:
                continue
    print(f"[Fault Injection Events] Found {len(times)} events.")
    if not times:
        print("No fault injection events to plot, skipping.")
        return
    df = pd.DataFrame({"timestamp": times, "type": types})
    plt.figure(figsize=(10, 3))
    sns.histplot(data=df, x="timestamp", hue="type", multiple="stack", bins=50)
    plt.title("Fault Injection Events Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fault_injection_events.png"))
    plt.close()

def plot_end_to_end_results(eval_dir, outdir):
    files = glob.glob(os.path.join(eval_dir, "*.json"))
    count = 0
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
            if isinstance(d, dict) and "results" in d:
                df = pd.DataFrame(d["results"])
                for col in df.columns:
                    if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
                        if df[col].notnull().sum() == 0:
                            print(f"No data for end-to-end metric {col}, skipping plot.")
                            continue
                        print(f"[End-to-End Results] Plotting {col} from {os.path.basename(f)} ({df[col].notnull().sum()} points)")
                        plt.figure(figsize=(8, 3))
                        plt.plot(df["timestamp"], df[col])
                        plt.title(f"{col} over time ({os.path.basename(f)})")
                        plt.tight_layout()
                        plt.savefig(os.path.join(outdir, f"end2end_{col}_{os.path.basename(f)}.png"))
                        plt.close()
                        count += 1
        except Exception:
            continue
    print(f"[End-to-End Results] Plotted {count} metrics from end-to-end test results.")

def main():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    outdir = os.path.join(base, "evaluation", "plots")
    os.makedirs(outdir, exist_ok=True)

    print("Generating anomaly and fault timeline plot...")
    plot_anomaly_timeline(os.path.join(base, "anomalies"), os.path.join(base, "faults"), outdir)

    print("Generating graph evolution plot...")
    plot_graph_evolution(os.path.join(base, "graphs"), outdir)

    print("Generating evaluation metrics plots...")
    plot_evaluation_metrics(os.path.join(base, "evaluation"), outdir)

    print("Generating service health timeline plot...")
    plot_service_health(os.path.join(base, "metrics"), outdir)

    print("Generating fault injection events plot...")
    plot_fault_injection_events(os.path.join(base, "fault_injection"), outdir)

    print("Generating end-to-end test results plots...")
    plot_end_to_end_results(os.path.join(base, "evaluation"), outdir)

    print(f"All plots saved to {outdir}")

if __name__ == "__main__":
    main()