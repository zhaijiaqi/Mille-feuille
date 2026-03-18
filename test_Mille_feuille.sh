#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./test_Mille_feuille.sh [matrix_set.csv]
#   不传参时默认读取本目录下的 valid_matrix_set.csv
#
# 说明：
# - 模仿 test_PETSc.sh 和 baselines/cusparse/test_cuSPARSE.sh
# - 逐行读取 csv 的 Name 列（第3列），在 MATRIX_ROOT 下查找对应的 <Name>.mtx
# - 同时执行 main-cg 和 main-cg-mixed，分别写入两个 csv
#
# 环境变量：
#   MATRIX_ROOT      矩阵根目录，默认 /data/matrix
#   MAX_IT           最大迭代次数，默认 10000
#   TIMEOUT_LIMIT    单矩阵超时，默认 4m
#   MAX_MATS         最多测试矩阵数，0 表示不限制
#   OUT_CSV_CG       main-cg 输出，默认 data/mille_feuille_cg.csv
#   OUT_CSV_MIXED    main-cg-mixed 输出，默认 data/mille_feuille_cg_mixed.csv

max_it="${MAX_IT:-10000}"
timeout_limit="${TIMEOUT_LIMIT:-4m}"
matrix_root="${MATRIX_ROOT:-/data/matrix}"
max_mats="${MAX_MATS:-0}"
out_csv_cg="${OUT_CSV_CG:-mille_feuille_cg.csv}"
out_csv_mixed="${OUT_CSV_MIXED:-mille_feuille_cg_mixed.csv}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 未传参时默认使用脚本同目录下的 valid_matrix_set.csv
input="${1:-${script_dir}/valid_matrix_set.csv}"

if [[ ! -f "${input}" ]]; then
  echo "ERROR: input csv not found: ${input}" >&2
  exit 1
fi

# 对每个矩阵同时运行 cg 和 cg_mixed，分别写入各自 csv
# $1=mtx, $2=it_cg (max_iter), $3=it_mix (max_iter_mix，供 main-cg-mixed 用)
run_matrix() {
  local mtx="$1"
  local it_cg="${2:-${max_it}}"
  local it_mix="${3:-$(( (${2:-${max_it}} * 125 + 99) / 100 ))}"  # 无 max_iter_mix 时用 ceil(max_iter*1.25)
  local bin_cg="${script_dir}/main-cg"
  local bin_mix="${script_dir}/main-cg-mixed"
  [[ -x "${bin_cg}" ]] || { echo "ERROR: ${bin_cg} not found, run make cg" >&2; return 1; }
  [[ -x "${bin_mix}" ]] || { echo "ERROR: ${bin_mix} not found, run make cg-mix" >&2; return 1; }
  echo "  [cg] max_iter=${it_cg}"
  timeout -s 9 "${timeout_limit}" "${bin_cg}" "${mtx}" "${it_cg}" "${out_csv_cg}" || true
  echo "  [cg-mixed] max_iter_mix=${it_mix}"
  timeout -s 9 "${timeout_limit}" "${bin_mix}" "${mtx}" "${it_mix}" "${out_csv_mixed}" || true
}

# 预检查二进制
[[ -x "${script_dir}/main-cg" ]] || { echo "ERROR: main-cg not found, run make cg" >&2; exit 1; }
[[ -x "${script_dir}/main-cg-mixed" ]] || { echo "ERROR: main-cg-mixed not found, run make cg-mix" >&2; exit 1; }

cd "${script_dir}"
mkdir -p data
mkdir -p "$(dirname "${out_csv_cg}")"
mkdir -p "$(dirname "${out_csv_mixed}")"

# 读 header，然后逐行处理（支持 max_iter、max_iter_mix 列）
{
  read -r header
  i=0
  while IFS=',' read -r id group name rows cols entries max_iter_row max_iter_mix_row; do
    [[ -z "${name}" ]] && continue
    if [[ "${max_mats}" != "0" ]] && [[ "${i}" -ge "${max_mats}" ]]; then
      break
    fi
    # 若环境变量 MAX_IT 已设置则优先用 MAX_IT；否则用 csv 中的 max_iter；最后用默认 10000
    if [[ -n "${MAX_IT:-}" ]]; then
      it="${max_it}"
      it_mix=$(( (it * 125 + 99) / 100 ))  # MAX_IT 覆盖时，it_mix = ceil(it*1.25)
    else
      it="${max_iter_row:-${max_it}}"
      it_mix="${max_iter_mix_row// /}"
      [[ -z "${it_mix}" ]] && it_mix=$(( (it * 125 + 99) / 100 ))  # 无 max_iter_mix 时用 ceil(it*1.25)
    fi
    it="${it// /}"

    direct="${matrix_root}/${name}.mtx"
    if [[ -f "${direct}" ]]; then
      echo "RUN ${direct} (max_iter=${it}, max_iter_mix=${it_mix})"
      run_matrix "${direct}" "${it}" "${it_mix}"
      i=$((i + 1))
      continue
    fi

    while IFS= read -r mtx; do
      [[ -z "${mtx}" ]] && continue
      echo "RUN ${mtx} (max_iter=${it}, max_iter_mix=${it_mix})"
      run_matrix "${mtx}" "${it}" "${it_mix}"
      i=$((i + 1))
      if [[ "${max_mats}" != "0" ]] && [[ "${i}" -ge "${max_mats}" ]]; then
        break 2
      fi
    done < <(find "${matrix_root}" -name "${name}.mtx" 2>/dev/null || true)
  done
} < "${input}"

echo "Done. Results: cg -> ${out_csv_cg}, cg-mixed -> ${out_csv_mixed}"
