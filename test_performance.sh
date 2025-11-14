bash test_need.sh
bash test_PETSc.sh
bash test_ginkgo.sh
python3 analy_cg_a100.py
python3 analy_bicg_a100.py
python3 analy.py
python3 convergence.py