"""
database.py
SQLite persistence layer for GoldenEye Detector - Evaluation Edition.

Each uploaded CSV becomes one "evaluation run" tagged with a year (parsed
from the filename). Runs are compared across years on the Trend page to
judge how well the model generalizes over time.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "goldeneye_eval.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            year TEXT,
            evaluated_at TEXT,
            total_rows INTEGER,
            evaluated_rows INTEGER,
            unmapped_rows INTEGER,
            has_ground_truth INTEGER,
            label_column TEXT,
            benign_count INTEGER,
            dos_count INTEGER,
            accuracy REAL,
            precision_ REAL,
            recall_ REAL,
            f1_ REAL,
            tp INTEGER, tn INTEGER, fp INTEGER, fn INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER,
            source_ip TEXT,
            source_port INTEGER,
            dest_ip TEXT,
            dest_port INTEGER,
            protocol TEXT,
            flow_duration REAL,
            true_label TEXT,
            prediction TEXT,
            correct INTEGER,
            probability REAL,
            features_json TEXT,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_flows_eval ON flows(evaluation_id)")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_flows_eval_pred
        ON flows(evaluation_id, prediction)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_flows_eval_correct
        ON flows(evaluation_id, correct)
    """)
    conn.commit()
    conn.close()


def save_evaluation(filename, year, result_df, feat_df, metrics):
    """Persist one evaluation run + its per-flow predictions. Returns evaluation id."""
    conn = get_conn()
    cur = conn.cursor()

    total = len(result_df)
    benign = int((result_df["Prediction"] == "BENIGN").sum())
    dos = int((result_df["Prediction"] == "DoS GoldenEye").sum())
    has_gt = metrics is not None
    evaluated_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    label_col = result_df.attrs.get("label_col")
    unmapped_n = result_df.attrs.get("unmapped_n", 0)

    cur.execute("""
        INSERT INTO evaluations
        (filename, year, evaluated_at, total_rows, evaluated_rows, unmapped_rows,
         has_ground_truth, label_column, benign_count, dos_count,
         accuracy, precision_, recall_, f1_, tp, tn, fp, fn)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, year, evaluated_at, total,
        metrics["evaluated_rows"] if has_gt else 0,
        unmapped_n, int(has_gt), label_col, benign, dos,
        metrics["accuracy"] if has_gt else None,
        metrics["precision"] if has_gt else None,
        metrics["recall"] if has_gt else None,
        metrics["f1"] if has_gt else None,
        metrics["tp"] if has_gt else None,
        metrics["tn"] if has_gt else None,
        metrics["fp"] if has_gt else None,
        metrics["fn"] if has_gt else None,
    ))
    evaluation_id = cur.lastrowid

    rows = []
    for i in range(total):
        r = result_df.iloc[i]
        feats = feat_df.iloc[i].to_dict()
        true_label = r["TrueLabel"] if pd_notna(r["TrueLabel"]) else None
        correct = None
        if true_label is not None:
            correct = int(true_label == r["Prediction"])
        rows.append((
            evaluation_id,
            str(r["Source IP"]), int(r["Source Port"]),
            str(r["Destination IP"]), int(r["Destination Port"]),
            str(r["Protocol"]), float(r["Flow Duration"]),
            true_label, str(r["Prediction"]), correct,
            float(r["Probability"]), json.dumps(feats, default=float),
        ))
    cur.executemany("""
        INSERT INTO flows
        (evaluation_id, source_ip, source_port, dest_ip, dest_port, protocol,
         flow_duration, true_label, prediction, correct, probability, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    return evaluation_id


def pd_notna(v):
    try:
        import pandas as pd
        return pd.notna(v)
    except Exception:
        return v is not None


def get_evaluations():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM evaluations ORDER BY id DESC").fetchall()
    conn.close()
    return rows


def get_evaluations_with_ground_truth_sorted_by_year():
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM evaluations WHERE has_ground_truth = 1 ORDER BY year ASC, id ASC
    """).fetchall()
    conn.close()
    return rows


def get_evaluation(evaluation_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    conn.close()
    return row


def get_flow(flow_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM flows WHERE id = ?", (flow_id,)).fetchone()
    conn.close()
    return row


def get_flows_filtered(evaluation_id, result_filter="Semua", search="", limit=10, offset=0):
    """
    result_filter:
        'Semua'
        'BENIGN' | 'DoS GoldenEye'        -> berdasarkan hasil prediksi
        'Prediksi Benar' | 'Prediksi Salah' -> berdasarkan kecocokan dengan label asli
    search: substring match on source/destination IP
    """
    conn = get_conn()
    where = ["evaluation_id = ?"]
    params = [evaluation_id]

    if result_filter == "BENIGN":
        where.append("prediction = 'BENIGN'")
    elif result_filter == "DoS GoldenEye":
        where.append("prediction = 'DoS GoldenEye'")
    elif result_filter == "Prediksi Benar":
        where.append("correct = 1")
    elif result_filter == "Prediksi Salah":
        where.append("correct = 0")

    if search:
        where.append("(source_ip LIKE ? OR dest_ip LIKE ?)")
        like = f"%{search}%"
        params.extend([like, like])

    where_sql = " AND ".join(where)
    total = conn.execute(
        f"SELECT COUNT(*) AS c FROM flows WHERE {where_sql}", params
    ).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM flows WHERE {where_sql} ORDER BY id ASC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()
    return rows, total


def delete_evaluation(evaluation_id):
    """Menghapus data dari tabel flows dan evaluations berdasarkan evaluation_id"""
    conn = get_conn()
    try:
        # Hapus data flow yang terikat dengan evaluasi ini terlebih dahulu
        conn.execute("DELETE FROM flows WHERE evaluation_id = ?", (evaluation_id,))
        
        # Hapus data utama evaluasinya
        conn.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
