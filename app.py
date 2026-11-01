"""
main.py (Flask app)
GoldenEye Detector - Evaluation Edition.

Evaluates whether a Random Forest DoS-GoldenEye model, trained with a
feature selection method combining Gini importance and Cohen's d effect
size, still generalizes to BENIGN / DoS GoldenEye traffic captured in
different years, including under feature perturbation.

Run with:  python app.py   (or  flask --app app run)
"""

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, Response

import database as db
from detection import (
    run_detection, compute_metrics, extract_year, load_model_and_features
)

BASE_DIR = Path(__file__).parent

app = Flask(__name__)
db.init_db()

def fmt_pct(value, base_decimals=2):

    if value is None:
        return "-"
    pct = value * 100
    rounded = round(pct, base_decimals)
    if rounded >= 100.0 and pct < 100.0:
        for extra in range(base_decimals + 1, 7):
            r = round(pct, extra)
            if r < 100.0:
                return f"{pct:.{extra}f}%"
        return f"{pct:.6f}%"
    return f"{pct:.{base_decimals}f}%"


app.jinja_env.filters["pct"] = fmt_pct

# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html", active="dashboard",
        error=request.args.get("error"), warning=request.args.get("warning"),
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return redirect(url_for("dashboard", error="Silakan upload file CSV terlebih dahulu."))
    if not file.filename.lower().endswith(".csv"):
        return redirect(url_for("dashboard", error="Hanya file .csv yang diperbolehkan."))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return redirect(url_for("dashboard", error=f"Gagal membaca CSV: {e}"))

    if df.empty:
        return redirect(url_for("dashboard", error="File CSV kosong."))

    try:
        result_df, feat_df, missing, label_col, unmapped_n = run_detection(df)
    except Exception as e:
        return redirect(url_for("dashboard", error=f"Gagal menjalankan model: {e}"))

    metrics = compute_metrics(result_df) if label_col is not None else None
    year = extract_year(file.filename)

    result_df.attrs["label_col"] = label_col
    result_df.attrs["unmapped_n"] = unmapped_n

    evaluation_id = db.save_evaluation(file.filename, year, result_df, feat_df, metrics)

    warn = None
    if missing:
        warn = f"{len(missing)} fitur model tidak ditemukan di CSV dan diisi 0."
    if label_col is None:
        w2 = "Tidak ditemukan kolom Label di CSV — hanya prediksi yang ditampilkan, akurasi tidak dihitung."
        warn = f"{warn} {w2}" if warn else w2
    elif unmapped_n:
        w3 = f"{unmapped_n} baris memiliki label yang tidak dikenali (bukan BENIGN/DoS GoldenEye) dan diabaikan dari perhitungan akurasi."
        warn = f"{warn} {w3}" if warn else w3

    return redirect(url_for("result", evaluation_id=evaluation_id, warning=warn))


# ----------------------------------------------------------------------
# Evaluation Result
# ----------------------------------------------------------------------
@app.route("/result/<int:evaluation_id>")
def result(evaluation_id):
    ev = db.get_evaluation(evaluation_id)
    if ev is None:
        return redirect(url_for("history", error="Data evaluasi tidak ditemukan."))

    result_filter = request.args.get("filter", "Semua")
    search = request.args.get("search", "")
    rows = request.args.get("rows", 10, type=int)
    if rows not in (5, 10, 25, 50):
        rows = 10
    page = max(1, request.args.get("page", 1, type=int))
    offset = (page - 1) * rows

    flows, total = db.get_flows_filtered(
        evaluation_id, result_filter=result_filter, search=search, limit=rows, offset=offset
    )
    total_pages = max(1, math.ceil(total / rows))
    if page > total_pages:
        page = total_pages
        offset = (page - 1) * rows
        flows, total = db.get_flows_filtered(
            evaluation_id, result_filter=result_filter, search=search, limit=rows, offset=offset
        )
    page_window = range(max(1, page - 2), min(total_pages, page + 2) + 1)

    return render_template(
        "result.html", active="detection", ev=ev, flows=flows, total=total,
        rows=rows, page=page, total_pages=total_pages, page_window=page_window,
        result_filter=result_filter, search=search,
        warning=request.args.get("warning"),
        start_index=offset + 1 if total else 0,
        end_index=min(offset + rows, total),
    )


# ----------------------------------------------------------------------
# Evaluasi (membuka hasil evaluasi terbaru tanpa redirect)
# ----------------------------------------------------------------------
@app.route("/evaluasi")
def evaluasi():
    evals = db.get_evaluations()
    if not evals:
        return render_template(
            "dashboard.html", active="dashboard",
            warning="Belum ada evaluasi tersimpan. Unggah dataset untuk memulai."
        )
    return result(evals[0]["id"])


# ----------------------------------------------------------------------
# Uji Ketahanan: perbandingan seleksi fitur Gini vs Metode Gabungan
# ----------------------------------------------------------------------
DAFTAR_LEVEL = [
    {"kode": "dasar", "nama": "Tanpa Gangguan", "nilai": 0.00,
     "kategori": "Acuan",
     "keterangan": "Kondisi awal tanpa gangguan, digunakan sebagai pembanding "
                   "untuk mengukur besarnya penurunan pada level berikutnya."},
    {"kode": "lv1", "nama": "Level 1", "nilai": 0.01,
     "kategori": "Sangat ringan",
     "keterangan": "Nilai fitur bergeser rata-rata 1% dari simpangan bakunya. "
                   "Setara variasi alami trafik jaringan sehari-hari."},
    {"kode": "lv2", "nama": "Level 2", "nilai": 0.02,
     "kategori": "Ringan",
     "keterangan": "Pergeseran 2% dari simpangan baku. Menyerupai kondisi "
                   "jaringan dengan latensi atau beban yang sedikit berbeda."},
    {"kode": "lv3", "nama": "Level 3", "nilai": 0.05,
     "kategori": "Sedang",
     "keterangan": "Pergeseran 5% dari simpangan baku. Menyerupai penyerang "
                   "yang menyisipkan jeda acak antar paket."},
    {"kode": "lv4", "nama": "Level 4", "nilai": 0.10,
     "kategori": "Berat",
     "keterangan": "Pergeseran 10% dari simpangan baku. Menyerupai penyamaran "
                   "aktif dengan mengubah ukuran dan interval paket."},
    {"kode": "lv5", "nama": "Level 5", "nilai": 0.20,
     "kategori": "Sangat berat",
     "keterangan": "Pergeseran 20% dari simpangan baku. Kondisi ekstrem untuk "
                   "menguji batas kemampuan model."},
]

LEVEL_NOISE = [lv["nilai"] for lv in DAFTAR_LEVEL]

PETA_LABEL_SERANGAN = ("dos goldeneye", "dos attacks-goldeneye")


def muat_pasangan(nama_model, nama_fitur):
    """Memuat sepasang berkas model dan daftar fiturnya, atau (None, None)."""
    import joblib

    for folder in (BASE_DIR / "models", BASE_DIR):
        berkas_model = folder / nama_model
        berkas_fitur = folder / nama_fitur
        if berkas_model.exists() and berkas_fitur.exists():
            return joblib.load(berkas_model), list(joblib.load(berkas_fitur))

    return None, None


def muat_semua_model():
    """
    Memuat ketiga model yang dibandingkan pada halaman uji ketahanan.

    1. Gini       : seleksi fitur konvensional, tanpa augmentasi
    2. Gabungan   : seleksi fitur dua kriteria, tanpa augmentasi
    3. Augmentasi : seleksi fitur dua kriteria disertai augmentasi gangguan

    Model yang berkasnya tidak tersedia akan dilewati.
    """
    daftar = []

    model_gini, fitur_gini = muat_pasangan(
        "random_forest_gini.pkl", "selected_features_gini.pkl")
    if model_gini is not None:
        daftar.append({
            "kode": "gini", "nama": "Gini",
            "judul": "Seleksi konvensional",
            "warna": "secondary",
            "model": model_gini, "fitur": fitur_gini,
        })

    model_gab, fitur_gab = muat_pasangan(
        "random_forest_gabungan.pkl", "selected_features_gabungan.pkl")
    if model_gab is not None:
        daftar.append({
            "kode": "gabungan", "nama": "Gabungan",
            "judul": "Seleksi dua kriteria",
            "warna": "info",
            "model": model_gab, "fitur": fitur_gab,
        })

    model_final, fitur_final = load_model_and_features()
    daftar.append({
        "kode": "augmentasi", "nama": "Augmentasi",
        "judul": "Dua kriteria + augmentasi",
        "warna": "primary",
        "model": model_final, "fitur": list(fitur_final),
    })

    return daftar


def muat_imputer():
    """
    Memuat imputer yang dipakai saat pelatihan, agar penanganan nilai kosong
    di halaman ini sama persis dengan proses deteksi pada halaman lain.
    Mengembalikan None bila berkasnya tidak tersedia.
    """
    import joblib

    for folder in (BASE_DIR / "models", BASE_DIR):
        berkas = folder / "imputer_final.pkl"
        if berkas.exists():
            return joblib.load(berkas)
    return None


def siapkan_matriks(df, kolom_fitur, imputer=None, kolom_latih=None):
    """
    Menyusun matriks fitur dari CSV mentah sesuai daftar kolom yang diminta.
    Bila imputer hasil pelatihan tersedia, nilai kosong diisi menggunakan
    statistik dari data latih, bukan dari data uji.
    """
    if imputer is not None and kolom_latih is not None:
        X_penuh = df.reindex(columns=kolom_latih)
        X_penuh = X_penuh.apply(pd.to_numeric, errors="coerce")
        X_penuh = X_penuh.replace([np.inf, -np.inf], np.nan)
        X_penuh = pd.DataFrame(imputer.transform(X_penuh), columns=kolom_latih)
        return X_penuh.reindex(columns=kolom_fitur).fillna(0.0)

    X = df.reindex(columns=kolom_fitur)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median()).fillna(0.0)


def hitung_metrik_biner(y_benar, y_prediksi):
    tp = int(((y_prediksi == 1) & (y_benar == 1)).sum())
    tn = int(((y_prediksi == 0) & (y_benar == 0)).sum())
    fp = int(((y_prediksi == 1) & (y_benar == 0)).sum())
    fn = int(((y_prediksi == 0) & (y_benar == 1)).sum())

    akurasi = (tp + tn) / len(y_benar) if len(y_benar) else 0.0
    presisi = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * presisi * recall / (presisi + recall) if (presisi + recall) else 0.0

    return {"accuracy": akurasi, "precision": presisi, "recall": recall,
            "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def jalankan_uji_noise(model, X, y, nama):
    """Menguji satu model pada seluruh tingkat gangguan."""
    simpangan = X.std().replace(0, 1).values
    kolom_nonnegatif = X.columns[(X.min() >= 0)]
    pengacak = np.random.RandomState(42)

    hasil = []
    for level in LEVEL_NOISE:
        X_uji = X + pengacak.normal(0, 1, X.shape) * simpangan * level
        X_uji[kolom_nonnegatif] = X_uji[kolom_nonnegatif].clip(lower=0)

        prediksi = model.predict(X_uji)
        if prediksi.dtype == object:
            p = np.array([1 if str(v).lower() in PETA_LABEL_SERANGAN else 0
                          for v in prediksi])
        else:
            p = prediksi.astype(int)

        baris = {"metode": nama, "level": level,
                 "level_persen": f"{level * 100:.0f}%"}
        baris.update(hitung_metrik_biner(y, p))
        hasil.append(baris)

    return hasil


@app.route("/ketahanan", methods=["GET", "POST"])
def ketahanan():
    """
    Menguji ketahanan model terhadap gangguan nilai fitur, sekaligus
    membandingkan tiga tahap perbaikan: seleksi Gini konvensional, seleksi
    dua kriteria, dan seleksi dua kriteria yang disertai augmentasi gangguan.
    """
    daftar_model = muat_semua_model()

    info_model = [
        {"kode": m["kode"], "nama": m["nama"], "judul": m["judul"],
         "warna": m["warna"], "n_fitur": len(m["fitur"])}
        for m in daftar_model
    ]

    fitur_akhir = daftar_model[-1]["fitur"]
    model_gini = next((m for m in daftar_model if m["kode"] == "gini"), None)

    konteks = {
        "active": "ketahanan",
        "daftar_level": DAFTAR_LEVEL,
        "info_model": info_model,
        "jumlah_model": len(daftar_model),
        "n_fitur_gabungan": len(fitur_akhir),
        "ada_pembanding": model_gini is not None,
        "n_fitur_gini": len(model_gini["fitur"]) if model_gini else 0,
        "irisan": len(set(fitur_akhir) & set(model_gini["fitur"])) if model_gini else 0,
        "fitur_bersama": sorted(set(fitur_akhir) & set(model_gini["fitur"])) if model_gini else [],
        "hanya_gini": sorted(set(model_gini["fitur"]) - set(fitur_akhir)) if model_gini else [],
        "hanya_gabungan": sorted(set(fitur_akhir) - set(model_gini["fitur"])) if model_gini else [],
        "hasil": None,
        "error": request.args.get("error"),
    }

    if request.method == "GET":
        return render_template("ketahanan.html", **konteks)

    berkas = request.files.get("file")
    if berkas is None or berkas.filename == "":
        konteks["error"] = "Silakan pilih file CSV terlebih dahulu."
        return render_template("ketahanan.html", **konteks)

    if not berkas.filename.lower().endswith(".csv"):
        konteks["error"] = "Hanya file .csv yang diperbolehkan."
        return render_template("ketahanan.html", **konteks)

    try:
        df = pd.read_csv(berkas, low_memory=False)
    except Exception as e:
        konteks["error"] = f"Gagal membaca CSV: {e}"
        return render_template("ketahanan.html", **konteks)

    df.columns = df.columns.str.strip()

    kolom_label = next((c for c in df.columns if c.strip().lower() == "label"), None)
    if kolom_label is None:
        konteks["error"] = ("File tidak memiliki kolom Label. Pengujian ketahanan "
                            "memerlukan label asli sebagai acuan perhitungan metrik.")
        return render_template("ketahanan.html", **konteks)

    df = df[df[kolom_label].astype(str).str.strip().str.lower() != "label"]

    # Pemetaan label ke tiga keadaan agar konsisten dengan halaman Evaluasi:
    #   1  -> serangan DoS GoldenEye
    #   0  -> lalu lintas normal (BENIGN)
    #  -1  -> label lain, dikeluarkan dari perhitungan metrik
    label_bersih = df[kolom_label].astype(str).str.strip().str.lower()

    def petakan_label(nilai):
        if nilai in PETA_LABEL_SERANGAN:
            return 1
        if nilai.startswith("benign"):
            return 0
        return -1

    y_penuh = label_bersih.apply(petakan_label).values
    mask_dikenali = y_penuh != -1
    jumlah_diabaikan = int((~mask_dikenali).sum())

    df = df[mask_dikenali]
    y = y_penuh[mask_dikenali]

    if len(y) == 0 or len(np.unique(y)) < 2:
        konteks["error"] = ("File tidak memuat kedua kelas yang diperlukan, "
                            "sehingga metrik tidak dapat dihitung.")
        return render_template("ketahanan.html", **konteks)

    BATAS_SAMPEL = 500000
    diambil_sampel = len(df) > BATAS_SAMPEL

    if diambil_sampel:
        indeks = np.random.RandomState(42).choice(len(df), BATAS_SAMPEL, replace=False)
        df = df.iloc[indeks]
        y = y[indeks]

    kolom_hilang = [c for c in fitur_akhir if c not in df.columns]
    if len(kolom_hilang) == len(fitur_akhir):
        konteks["error"] = "Tidak ada satu pun kolom fitur model yang ditemukan pada file ini."
        return render_template("ketahanan.html", **konteks)

    imputer = muat_imputer()
    kolom_latih = list(imputer.feature_names_in_) if (
        imputer is not None and hasattr(imputer, "feature_names_in_")) else None

    hasil = []
    for m in daftar_model:
        X = siapkan_matriks(df, m["fitur"], imputer, kolom_latih)
        hasil += jalankan_uji_noise(m["model"], X, y, m["nama"])

    nama_pertama = daftar_model[0]["nama"]
    nama_terakhir = daftar_model[-1]["nama"]

    ringkas = []
    for info in DAFTAR_LEVEL:
        baris = dict(info)
        baris["level_persen"] = f"{info['nilai'] * 100:.0f}%"
        baris["metrik"] = {}

        for m in daftar_model:
            cocok = [h for h in hasil
                     if h["metode"] == m["nama"] and h["level"] == info["nilai"]]
            baris["metrik"][m["nama"]] = cocok[0] if cocok else None

        awal = baris["metrik"].get(nama_pertama)
        akhir = baris["metrik"].get(nama_terakhir)
        if awal and akhir and awal["f1"] > 0:
            baris["rasio"] = akhir["f1"] / awal["f1"]

        ringkas.append(baris)

    konteks.update({
        "hasil": hasil,
        "ringkas": ringkas,
        "n_sampel": len(y),
        "proporsi_serangan": float(y.mean()),
        "nama_berkas": berkas.filename,
        "kolom_hilang": kolom_hilang,
        "jumlah_diabaikan": jumlah_diabaikan,
        "diambil_sampel": diambil_sampel,
        "ada_imputer": imputer is not None,
    })

    return render_template("ketahanan.html", **konteks)


# ----------------------------------------------------------------------
# Flow detail partial (AJAX, loaded into a Bootstrap modal)
# ----------------------------------------------------------------------
@app.route("/api/flow/<int:flow_id>")
def flow_detail(flow_id):
    import json
    flow = db.get_flow(flow_id)
    if flow is None:
        return "<div class='p-3'>Data flow tidak ditemukan.</div>", 404

    feats = {}
    if flow["features_json"]:
        try:
            feats = json.loads(flow["features_json"])
        except Exception:
            feats = {}
    feature_items = list(feats.items())[:14]

    return render_template("partials/flow_detail.html", flow=flow, feature_items=feature_items)


# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------
@app.route("/history")
def history():
    evals = db.get_evaluations()
    return render_template("history.html", active="history", evaluations=evals)


@app.route("/delete/<int:evaluation_id>", methods=["POST"])
def delete_evaluation(evaluation_id):
    """Menghapus data evaluasi beserta flow terkait berdasarkan ID"""
    try:
        db.delete_evaluation(evaluation_id)
    except Exception as e:
        print(f"Gagal menghapus data: {e}")
    
    return redirect(url_for("history"))

# ----------------------------------------------------------------------
# About
# ----------------------------------------------------------------------
@app.route("/about")
def about():
    model, features = load_model_and_features()

    importances = []
    if hasattr(model, "feature_importances_"):
        pairs = sorted(zip(features, model.feature_importances_),
                       key=lambda p: p[1], reverse=True)[:10]
        max_imp = pairs[0][1] if pairs else 1.0
        importances = [
            {"name": name, "value": float(val),
             "pct": round(float(val) / max_imp * 100, 1)}
            for name, val in pairs
        ]

    return render_template(
        "about.html", active="about",
        n_estimators=model.get_params().get("n_estimators", "-"),
        max_depth=model.get_params().get("max_depth", "-"),
        n_features=len(features),
        daftar_fitur=list(features),
        importances=importances,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
