input="cg_matrix_set.csv"
{
  read
  i=1
  while IFS=',' read -r data a b each_nnz
  do
    for data1 in `find "./matrix" -name "$data.mtx"`
    do
      timeout -s 9 1m ./pre_time $data1 $each_nnz
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
      timeout -s 9 1m ./pre_time $data1 $each_nnz
    done
    i=`expr $i + 1`
  done 
} < "$input"

python3 pre_time.py