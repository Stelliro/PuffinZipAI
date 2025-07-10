@echo off
rem This script runs the benchmark data set generator.
rem The generated files will be placed in the location defined in your config.py,
rem typically under /data/benchmark_sets/generated_numeric_benchmark_v1/

echo --- Starting PuffinZipAI Benchmark Data Generator ---
python -m puffinzip_ai.utils.benchmark_generator

echo --- Benchmark generation finished. ---
pause