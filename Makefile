# Top-level entry points.  `make help` lists them.
PY ?= python3

help:
	@echo "make lib        build csrc/ -> src/radiotel/correlator/libfxcorr.so and the streaming tools"
	@echo "make test       run the test suite"
	@echo "make examples   run the four examples (writes figs/)"
	@echo "make bench      run all benchmarks (writes benchmarks/results/)"
	@echo "make figures    regenerate benchmark figures + docs/benchmark_tables.md from results"
	@echo "make report     build report/report.pdf"
	@echo "make all        lib, test, examples, figures, report"
	@echo "make quick      fast smoke run (what CI does)"

lib:
	$(MAKE) -C csrc

test: lib
	$(PY) -m pytest -q

examples: lib
	$(PY) examples/01_two_element_fringe.py
	$(PY) examples/02_campus_array_imaging.py
	$(PY) examples/03_optimize_layout.py
	$(PY) examples/04_calibration.py

bench: lib
	$(PY) benchmarks/bench_correlator.py
	$(PY) benchmarks/bench_phase.py
	benchmarks/bench_stream.sh

figures:
	$(PY) benchmarks/make_figures.py

report: figures
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex >/dev/null && echo "report/report.pdf"

quick: lib
	$(PY) -m pytest -q
	for f in examples/0*.py; do $(PY) $$f --quick --out out/figs || exit 1; done
	$(PY) benchmarks/bench_correlator.py --quick
	benchmarks/bench_stream.sh --quick

all: lib test examples figures report

clean:
	$(MAKE) -C csrc clean
	rm -rf out
	cd report && latexmk -C >/dev/null 2>&1 || true

.PHONY: help lib test examples bench figures report quick all clean
