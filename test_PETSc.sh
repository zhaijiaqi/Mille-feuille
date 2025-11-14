input="cg_matrix_set.csv"
{
  read
  i=1
  while IFS=',' read -r data a b each_nnz
  do
    for data1 in `find "./matrix" -name "$data.mtx"`
    do
      echo -n $data1
      timeout -s 9 4m ./petsc-main/src/ksp/ksp/tutorials/test_cg $data1 -ksp_max_it 100 -ksp_monitor -ksp_type cg -mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0 -pc_type none -ksp_norm_type unpreconditioned
    done
    i=`expr $i + 1`
  done 
} < "$input"

input="bicg_matrix_set.csv"
{
  read
  i=1
  while IFS=',' read -r data a b each_nnz
  do
    for data1 in `find "./matrix" -name "$data.mtx"`
    do
      echo -n $data1
      timeout -s 9 4m ./petsc-main/src/ksp/ksp/tutorials/test_bicg $data1 -ksp_max_it 100 -ksp_monitor -ksp_type bicg -mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0 -pc_type none -ksp_norm_type unpreconditioned
    done
    i=`expr $i + 1`
  done 
} < "$input"