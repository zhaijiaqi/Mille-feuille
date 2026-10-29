# 这段代码批量读取cg_matrix_set.csv和bicg_matrix_set.csv中的矩阵名称，依次查找对应.mtx文件，
# 然后分别运行相关测试程序（如Mille_cg_ad、cg_cusparse等）对这些矩阵进行4分钟超时的性能测试，并输出结果。
input="cg_matrix_set.csv"
{
  read
  i=1
  while IFS=',' read -r data a b each_nnz
  do
    for data1 in `find "./matrix" -name "$data.mtx"`
    do
      echo -n $data1
      timeout -s 9 4m ./Mille_cg_ad $data1 $each_nnz
      timeout -s 9 4m ./cg_cusparse $data1 
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
      timeout -s 9 4m ./Mille_bicg_ad $data1 $each_nnz
      timeout -s 9 4m ./bicg_cusparse $data1 
    done
    i=`expr $i + 1`
  done 
} < "$input"