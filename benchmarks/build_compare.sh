#!/bin/bash
# Builds a static Highway VQSort library for a single ISA target (no runtime dispatch).
# usage: build_compare.sh <outdir> <march>     e.g.  build_compare.sh hwy_avx2 haswell
#        HWY=/path/to/google/highway ./build_compare.sh hwy_avx512 skylake-avx512
set -e
HWY=${HWY:-$HOME/highway}; out=$1; march=$2
mkdir -p "$out" && cd "$out"
SRC="hwy/targets.cc hwy/per_target.cc hwy/print.cc hwy/abort.cc hwy/timer.cc hwy/os_rng.cc hwy/contrib/sort/vqsort.cc"
for t in u16 i16 u32 i32 u64 i64 f32 f64; do SRC="$SRC hwy/contrib/sort/vqsort_${t}a.cc hwy/contrib/sort/vqsort_${t}d.cc"; done
for f in $SRC; do
  g++ -std=c++17 -O3 -march=$march -DHWY_COMPILE_ONLY_STATIC=1 -I"$HWY" -c "$HWY/$f" -o "$(echo $f | tr / _).o"
done
ar rcs libhwy_sort.a *.o
echo "built $out/libhwy_sort.a"
