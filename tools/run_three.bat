@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
set TG_RUN_CUDA_TESTS=1
C:\Python311\python.exe -X utf8 -m pytest tests/test_hygiene.py::test_no_crlf tests/test_ops_bundle.py::test_load_picks_python_on_cpu tests/test_version.py -q --tb=short
