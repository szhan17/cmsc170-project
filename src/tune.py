"""Issue #26: tune model settings on the fixed validation period.

Train: 2001-2017. Validation: 2018-2021. make_splits also returns the test
years, but this script never uses or scores them. Run from the project root:
    python src/tune.py

The main selection metric is validation F1 for the rainy class (label 1).
All scaler statistics and Naive Bayes bin cut points are learned from train only.
"""
import csv, json, os, sys, time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_scale import make_splits
from knn import KNNClassifier
from naive_bayes import NaiveBayesClassifier
from perceptron import AveragedPerceptron
from metrics import all_metrics

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS=os.path.join(ROOT,"results")
TUNE_DIR=os.path.join(RESULTS,"tuning")

# Deliberately compact, documented search grids.
KNN_GRID=list(product([21,41,61,81],[1.0,2.0,3.0]))
NB_GRID=list(product([5,8,10,12],[0.1,0.5,1.0,2.0]))
PERCEPTRON_GRID=list(product([5,10,20,30,50],[0.01,0.05,0.1,0.5]))
SEED=42


def evaluate(y,pred,method,settings,elapsed):
    m=all_metrics(y,pred)
    row={"method":method,**settings,"accuracy":m["accuracy"],"precision":m["precision"],
         "recall":m["recall"],"f1":m["f1"],"TP":m["TP"],"FP":m["FP"],
         "FN":m["FN"],"TN":m["TN"],"run_time_seconds":elapsed}
    return row


def choose_best(rows):
    # Primary criterion: F1. Deterministic secondary criteria: recall then accuracy.
    return max(rows,key=lambda r:(r["f1"],r["recall"],r["accuracy"]))


def save_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    os.makedirs(TUNE_DIR,exist_ok=True)
    # Only these train/validation splits are returned/used. Do not access test data.
    mm=make_splits(save=False,scaling="minmax")
    zs=make_splits(save=False,scaling="zscore")
    y_val=mm["y_val"]
    all_rows=[]; best_by_method={}

    # KNN grid: tune k and Minkowski exponent p on z-score features.
    rows=[]
    for k,p in KNN_GRID:
        start=time.perf_counter()
        model=KNNClassifier(k=k,p=p).fit(zs["X_train"],zs["y_train"])
        pred=model.predict(zs["X_val"])
        rows.append(evaluate(y_val,pred,"knn",{"k":k,"p":p,"scaling":"zscore"},time.perf_counter()-start))
    best=choose_best(rows);best_by_method["knn"]=best;all_rows+=rows
    save_csv(os.path.join(TUNE_DIR,"knn_tuning.csv"),rows)

    # Naive Bayes grid: bins and Laplace smoothing, using min-max features.
    rows=[]
    for bins,smoothing in NB_GRID:
        start=time.perf_counter()
        model=NaiveBayesClassifier(bins=bins,smoothing=smoothing).fit(mm["X_train"],mm["y_train"])
        pred=model.predict(mm["X_val"])
        rows.append(evaluate(y_val,pred,"naive_bayes",{"bins":bins,"smoothing":smoothing,"scaling":"minmax"},time.perf_counter()-start))
    best=choose_best(rows);best_by_method["naive_bayes"]=best;all_rows+=rows
    save_csv(os.path.join(TUNE_DIR,"naive_bayes_tuning.csv"),rows)

    # Perceptron grid: epochs and learning rate, with reproducible shuffling.
    rows=[]
    for epochs,lr in PERCEPTRON_GRID:
        start=time.perf_counter()
        model=AveragedPerceptron(n_epochs=epochs,seed=SEED,lr=lr).fit(zs["X_train"],zs["y_train"])
        pred=model.predict(zs["X_val"])
        rows.append(evaluate(y_val,pred,"perceptron",{"epochs":epochs,"learning_rate":lr,"seed":SEED,"scaling":"zscore"},time.perf_counter()-start))
    best=choose_best(rows);best_by_method["perceptron"]=best;all_rows+=rows
    save_csv(os.path.join(TUNE_DIR,"perceptron_tuning.csv"),rows)

    save_csv(os.path.join(TUNE_DIR,"all_settings.csv"),all_rows)
    with open(os.path.join(TUNE_DIR,"best_settings.json"),"w",encoding="utf-8") as f:
        json.dump(best_by_method,f,indent=2);f.write("\n")
    with open(os.path.join(TUNE_DIR,"best_settings.md"),"w",encoding="utf-8") as f:
        f.write("# Issue #26 — Best validation settings\n\nSelection criterion: highest validation F1 for rainy class; ties resolved by recall, then accuracy. Test years were not evaluated.\n\n")
        f.write("| Method | Best settings | Accuracy | Precision | Recall | F1 |\n|---|---|---:|---:|---:|---:|\n")
        for name,r in best_by_method.items():
            settings={k:v for k,v in r.items() if k in ("k","p","bins","smoothing","epochs","learning_rate","seed","scaling")}
            f.write(f"| {name} | `{settings}` | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n")
    print("Best settings by validation F1:")
    for name,r in best_by_method.items():
        print(f"{name}: settings={ {k:v for k,v in r.items() if k in ('k','p','bins','smoothing','epochs','learning_rate','seed','scaling')} } | F1={r['f1']:.4f} | Recall={r['recall']:.4f} | Accuracy={r['accuracy']:.4f}")
    print(f"\nSaved tuning tables and best-settings summary to results/tuning/ ({len(all_rows)} configurations).")

if __name__=="__main__": main()
