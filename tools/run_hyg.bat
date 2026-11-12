@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
C:\Python311\python.exe -X utf8 -m pytest tests/test_hygiene.py::test_no_crlf -q --tb=line
