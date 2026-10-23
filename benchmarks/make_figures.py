"""Turn benchmark CSVs into figures (figs/) and Markdown tables (docs/benchmark_tables.md)."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from _common import FIGS, RESULTS, ROOT

from radiotel import plotting
from radiotel.plotting import INK_2, REFERENCE, plt

ORDER = ["numpy-loop", "numpy-gemm", "torch-cpu", "torch-cuda", "c-scalar", "c-avx2", "c-avx2-omp"]


def read(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def correlator_figure(rows, threads):
    ok = [r for r in rows if r["status"] == "ok"]
    nchan = 1024 if any(r["nchan"] == "1024" for r in ok) else int(ok[0]["nchan"])
    by = defaultdict(list)
    for r in ok:
        if int(r["nchan"]) == nchan:
            by[r["backend"]].append((int(r["n_ant"]), float(r["t_xstep_s"]), float(r["gflops_x"]),
                                     float(r["rtf_xstep"]), float(r["rtf_total"])))
    n_samp = int(ok[0]["n_samples"])

    fig, ax = plt.subplots(1, 3 if threads else 2, figsize=(14 if threads else 10, 4.2))
    for b in ORDER:
        if b not in by:
            continue
        pts = sorted(by[b])
        n = [p[0] for p in pts]
        ax[0].plot(n, [p[1] * 1e3 for p in pts], **plotting.backend_style(b), label=b)
        ax[1].plot(n, [p[3] for p in pts], **plotting.backend_style(b), label=b)
    # N(N-1)/2 guide through the compute-bound c-avx2-omp point at N = 64 (the GPU is
    # launch-overhead-bound at small N, so anchoring on it would misplace the guide).
    ref = next((p for p in by.get("c-avx2-omp", []) if p[0] == 64), None)
    if ref:
        ns = np.array([2, 64])
        guide = ref[1] * 1e3 * (ns * (ns - 1)) / (64 * 63) * 2.5
        ax[0].plot(ns, guide, color=REFERENCE, linestyle="--", linewidth=1)
        ax[0].text(2.2, guide[0] * 1.5, "∝ N(N−1)/2", color=INK_2, fontsize=8.5, ha="left", va="bottom")
    ax[0].set(xscale="log", yscale="log", title=f"X-step time ({n_samp:,} samples/antenna, {nchan} ch)",
              xlabel="antennas N", ylabel="time (ms)")
    ax[1].axhline(1.0, color=REFERENCE, linewidth=1, linestyle="--")
    ax[1].text(2.1, 1.15, "real time at 2.4 MS/s", color=INK_2, fontsize=8.5)
    ax[1].set(xscale="log", yscale="log", title="Real-time factor (X-step)", xlabel="antennas N",
              ylabel="data duration / compute time")
    for a in ax[:2]:
        a.set_xticks([2, 4, 8, 16, 32, 64], labels=["2", "4", "8", "16", "32", "64"])
    handles, labels = ax[1].get_legend_handles_labels()

    if threads:
        t = sorted((int(r["threads"]), float(r["t_xstep_s"])) for r in threads)
        th = np.array([x[0] for x in t])
        sp = t[0][1] / np.array([x[1] for x in t])
        ax[2].plot(th, th, color=REFERENCE, linestyle="--", linewidth=1, label="ideal")
        ax[2].plot(th, sp, **plotting.backend_style("c-avx2-omp"), label="c-avx2-omp")
        ax[2].set(title=f"OpenMP scaling (N={threads[0]['n_ant']}, {threads[0]['nchan']} ch)",
                  xlabel="threads", ylabel="speed-up vs 1 thread")
        ax[2].legend(loc="upper left")
    # one shared legend under all panels (every panel uses the same backend colours)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return plotting.save(fig, FIGS / "correlator_scaling.png")


def phase_figure(snr, freq, runtime):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    names = list(dict.fromkeys(r["estimator"] for r in snr))
    for nme in names:
        pts = [(float(r["snr_db"]), float(r["rmse_rad"])) for r in snr if r["estimator"] == nme]
        ax[0].plot(*zip(*pts), **plotting.estimator_style(nme), label=nme)
    crlb = sorted({(float(r["snr_db"]), float(r["crlb_rmse_rad"])) for r in snr})
    ax[0].plot(*zip(*crlb), color=REFERENCE, linestyle="--", linewidth=1.2, label="Cramér–Rao bound")
    ax[0].set(yscale="log", title="Phase error vs SNR (N = 1024)", xlabel="per-channel SNR (dB)",
              ylabel="RMS error (rad)")
    ax[0].legend(loc="lower left", fontsize=8)

    for nme in names:
        pts = [(float(r["freq_hz"]), float(r["rmse_rad"])) for r in freq if r["estimator"] == nme]
        ax[1].plot(*zip(*pts), **plotting.estimator_style(nme, markersize=3.5), label=nme)
    ax[1].set(yscale="log", title="Error vs tone frequency (30 dB)", xlabel="frequency (Hz, fs = 1024 Hz)",
              ylabel="RMS error (rad)")

    for nme in names:
        pts = [(int(r["n_samples"]), float(r["time_s"]) * 1e3) for r in runtime if r["estimator"] == nme]
        ax[2].plot(*zip(*pts), **plotting.estimator_style(nme), label=nme)
    ax[2].set(xscale="log", yscale="log", title="Runtime vs signal length", xlabel="samples", ylabel="time (ms)")
    fig.tight_layout()
    return plotting.save(fig, FIGS / "phase_estimators.png")


def stream_figure(rows):
    un = sorted((int(r["n_ant"]), float(r["msps_per_ant"]), float(r["t_fstep_s"]), float(r["t_xstep_s"]),
                 float(r["elapsed_s"])) for r in rows if r["mode"] == "unlimited")
    if not un:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    n = [u[0] for u in un]
    ax[0].plot(n, [u[1] for u in un], **plotting.backend_style("c-avx2-omp"), label="corrd sustained")
    ax[0].axhline(2.4, color=REFERENCE, linestyle="--", linewidth=1)
    ax[0].text(n[0], 2.4 * 1.15, "2.4 MS/s per receiver (SDR)", color=INK_2, fontsize=8.5)
    ax[0].set(xscale="log", yscale="log", title="Streaming correlator throughput", xlabel="receivers",
              ylabel="MS/s per receiver")
    ax[0].set_xticks(n, labels=[str(x) for x in n])
    ticks = [1, 2, 5, 10, 20, 50]
    ax[0].set_yticks(ticks, labels=[str(t) for t in ticks])
    ax[0].minorticks_off()
    x = np.arange(len(un))
    f = np.array([u[2] / u[4] for u in un])
    xs = np.array([u[3] / u[4] for u in un])
    ax[1].bar(x, f, width=0.6, color=plotting.PALETTE[2], label="F-step (FFT)")
    ax[1].bar(x, xs, bottom=f, width=0.6, color=plotting.PALETTE[0], label="X-step (AVX2+OpenMP)",
              edgecolor=plotting.SURFACE, linewidth=1)
    ax[1].bar(x, 1 - f - xs, bottom=f + xs, width=0.6, color=plotting.GRID, label="I/O, unpack, sync",
              edgecolor=plotting.SURFACE, linewidth=1)
    ax[1].set(title="Where the time goes", xlabel="receivers", ylabel="fraction of wall time", ylim=(0, 1.0))
    ax[1].set_axisbelow(True)
    ax[1].set_xticks(x, labels=[str(u[0]) for u in un])
    ax[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return plotting.save(fig, FIGS / "stream_throughput.png")


def tables(corr, stream, layout):
    lines = ["# Benchmark tables", "",
             "Generated by `benchmarks/make_figures.py` from `benchmarks/results/*.csv`.", ""]
    ok = [r for r in corr if r["status"] == "ok"]
    if ok:
        lines += [f"Host: {ok[0]['cpu']} · commit `{ok[0]['git']}` · {int(ok[0]['n_samples']):,} samples per antenna", ""]
        for nchan in sorted({int(r["nchan"]) for r in ok}):
            sub = [r for r in corr if int(r["nchan"]) == nchan]
            ns = sorted({int(r["n_ant"]) for r in sub})
            lines += [f"## X-step real-time factor, {nchan} channels", "",
                      "| backend | " + " | ".join(f"N={n}" for n in ns) + " |",
                      "|---|" + "---:|" * len(ns)]
            for b in ORDER:
                cells = []
                for n in ns:
                    m = [r for r in sub if r["backend"] == b and int(r["n_ant"]) == n]
                    if not m:
                        cells.append("—")
                    elif m[0]["status"] != "ok":
                        cells.append("*pending*")
                    else:
                        cells.append(f"{float(m[0]['rtf_xstep']):.3g}")
                if any(c != "—" for c in cells):
                    lines.append(f"| {b} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += ["Real-time factor = data duration at 2.4 MS/s ÷ X-step compute time (> 1 keeps up). "
                  "`torch-cuda` rows were measured on a Colab GPU (`benchmarks/gpu_colab.ipynb`), all other rows on the "
                  "host above; *pending* = no GPU results file; — = skipped "
                  "because the estimated time exceeded 20 s.", ""]
    if stream:
        lines += ["## Streaming daemon (`corrd`)", "",
                  "| mode | receivers | MS/s per receiver | × real time | frames | lost | overruns | max lag (ms) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in stream:
            msps = float(r["msps_per_ant"])
            if r["mode"] == "paced":
                lag = f"{r['overruns']} | {float(r['max_lag_ms']):.1f}"
            else:
                lag = "— | —"
            lines.append(f"| {r['mode']} | {r['n_ant']} | {msps:.2f} | {msps / 2.4:.2f} | "
                         f"{r['frames']} | {r['lost']} | {lag} |")
        lines += ["", "Unlimited: senders as fast as back-pressure allows (= maximum sustainable rate). "
                  "Paced: 2.4 MS/s per receiver; an overrun is a frame delayed beyond a 32-frame (55 ms) "
                  "receiver buffer, i.e. data a real receiver would lose."]
        lines.append("")
    if layout:
        lines += ["## Array layout (u-v fill fraction)", "",
                  "| receivers | baselines | median random | best of random | annealed |",
                  "|---:|---:|---:|---:|---:|"]
        for r in layout:
            lines.append(f"| {r['n_ant']} | {r['n_baselines']} | {float(r['random_median']):.3f} | "
                         f"{float(r['random_best']):.3f} | {float(r['annealed']):.3f} |")
        r4 = next((r for r in layout if r.get("legacy_square")), None)
        if r4:
            lines += ["", f"Original 4-receiver layouts: line {float(r4['legacy_line']):.3f}, "
                          f"clustered line {float(r4['legacy_clustered_line']):.3f}, "
                          f"square {float(r4['legacy_square']):.3f}."]
        lines.append("")
    path = ROOT / "docs" / "benchmark_tables.md"
    path.write_text("\n".join(lines))
    return path


def latex_table(corr, nchan=1024):
    """Real-time factor table for the report (report/generated/correlator_table.tex)."""
    sub = [r for r in corr if int(r["nchan"]) == nchan]
    ns = sorted({int(r["n_ant"]) for r in sub})
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{X-step real-time factor at 2.4\,MS/s per receiver, " + str(nchan) +
             r" channels (higher is better; $>1$ keeps up with a live stream). "
             r"\texttt{torch-cuda} measured on a cloud GPU, other rows on the CPU host; "
             r"--: skipped (estimated $>$20\,s); pending: no GPU results available.}",
             r"\label{tab:corr}", r"\begin{tabular}{l" + "r" * len(ns) + "}", r"\toprule",
             "back end & " + " & ".join(f"$N={n}$" for n in ns) + r" \\", r"\midrule"]
    for b in ORDER:
        cells = []
        for n in ns:
            m = [r for r in sub if r["backend"] == b and int(r["n_ant"]) == n]
            if not m:
                cells.append("--")
            elif m[0]["status"] != "ok":
                cells.append(r"\textit{pending}")
            else:
                cells.append(f"{float(m[0]['rtf_xstep']):.3g}")
        if any(c != "--" for c in cells):
            lines.append(r"\texttt{" + b + "} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = ROOT / "report" / "generated" / "correlator_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# LaTeX macro names cannot contain digits.
WORD = {2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour"}


def latex_numbers(corr, threads, stream, layout, snr):
    """Key numbers as LaTeX macros (report/generated/numbers.tex), so the report
    always quotes the committed results."""
    out = {}

    def get(b, n, c, key):
        m = [r for r in corr if r["backend"] == b and int(r["n_ant"]) == n and int(r["nchan"]) == c
             and r["status"] == "ok"]
        return float(m[0][key]) if m else float("nan")

    ok = [r for r in corr if r["status"] == "ok"]
    if ok:
        out["CPUModel"] = ok[0]["cpu"].replace("(R)", "").replace("(TM)", "")
        out["NSamples"] = f"{int(ok[0]['n_samples']):,}".replace(",", "{,}")
        for b, tag in [("c-avx2-omp", "Omp"), ("c-avx2", "Avx"), ("c-scalar", "Scalar"),
                       ("numpy-gemm", "Gemm"), ("numpy-loop", "Loop"), ("torch-cpu", "Torch")]:
            for n in (8, 16, 64):
                v = get(b, n, 1024, "rtf_xstep")
                out[f"Rtf{tag}{WORD[n]}"] = f"{v:.3g}" if np.isfinite(v) else "--"
        out["GflopsOmpPeak"] = f"{max(float(r['gflops_x']) for r in ok if r['backend'] == 'c-avx2-omp'):.0f}"
        sp = get("c-scalar", 16, 1024, "t_xstep_s") / get("c-avx2", 16, 1024, "t_xstep_s")
        out["SpeedupAvxScalar"] = f"{sp:.1f}"
        sp = get("numpy-gemm", 64, 1024, "t_xstep_s") / get("c-avx2-omp", 64, 1024, "t_xstep_s")
        out["SpeedupOmpGemmSixtyFour"] = f"{sp:.1f}"
        gpu = [r for r in ok if r["backend"] == "torch-cuda"]
        if gpu:
            for n in (8, 64):
                out[f"RtfCuda{WORD[n]}"] = f"{get('torch-cuda', n, 1024, 'rtf_xstep'):.3g}"
            out["GflopsCudaPeak"] = f"{max(float(r['gflops_x']) for r in gpu):.0f}"
            sp = get("c-avx2-omp", 64, 1024, "t_xstep_s") / get("torch-cuda", 64, 1024, "t_xstep_s")
            out["SpeedupCudaOmpSixtyFour"] = f"{sp:.1f}"
            model = gpu[0]["device"].removeprefix("cuda").removeprefix(": ")
            out["GpuHost"] = model or "Colab"  # older CSVs did not record the card
    if threads:
        t = sorted((int(r["threads"]), float(r["t_xstep_s"])) for r in threads)
        out["ThreadsMax"] = str(t[-1][0])
        out["ThreadSpeedup"] = f"{t[0][1] / t[-1][1]:.1f}"
    for r in stream:
        n = WORD[int(r["n_ant"])]
        if r["mode"] == "unlimited":
            out[f"StreamMsps{n}"] = f"{float(r['msps_per_ant']):.2f}"
        else:
            out[f"StreamOverruns{n}"] = r["overruns"]
            out[f"StreamMaxLag{n}"] = f"{float(r['max_lag_ms']):.0f}"
    for r in layout:
        if r["n_ant"] == "8":
            out["LayoutAnnealEight"] = f"{float(r['annealed']):.3f}"
            out["LayoutRandBestEight"] = f"{float(r['random_best']):.3f}"
            out["LayoutRandMedEight"] = f"{float(r['random_median']):.3f}"
        if r.get("legacy_square"):
            out["LayoutLegacySquare"] = f"{float(r['legacy_square']):.3f}"
            out["LayoutLegacyLine"] = f"{float(r['legacy_line']):.3f}"
            out["LayoutAnnealFour"] = f"{float(r['annealed']):.3f}"
    for r in snr:
        if r["estimator"] == "hilbert_unwrap" and float(r["snr_db"]) == 0:
            out["HilbertUnwrapZeroDb"] = f"{float(r['rmse_rad']):.2f}"
        if r["estimator"] == "hilbert_cross" and float(r["snr_db"]) == 0:
            out["HilbertCrossZeroDb"] = f"{float(r['rmse_rad']):.3f}"
            out["CrlbZeroDb"] = f"{float(r['crlb_rmse_rad']):.3f}"
    path = ROOT / "report" / "generated" / "numbers.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("% generated by benchmarks/make_figures.py -- do not edit\n" +
                    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(out.items())))
    return path


def main():
    plotting.apply_theme()
    corr = read("correlator.csv") or read("correlator_quick.csv")
    gpu = read("correlator_gpu.csv")
    if gpu:  # GPU results from Colab replace the placeholders
        corr = [r for r in corr if r["backend"] != "torch-cuda"] + gpu
    threads = read("correlator_threads.csv")
    if corr:
        print(correlator_figure(corr, threads))
        print(latex_table(corr))
    snr, freq, rt = read("phase_vs_snr.csv"), read("phase_vs_freq.csv"), read("phase_runtime.csv")
    if snr:
        print(phase_figure(snr, freq, rt))
    stream = read("stream.csv")
    if stream:
        print(stream_figure(stream))
    print(tables(corr, stream, read("layout.csv")))
    print(latex_numbers(corr, threads, stream, read("layout.csv"), snr))


if __name__ == "__main__":
    main()
