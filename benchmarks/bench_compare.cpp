// bench_compare.cpp -- single-thread head-to-head: std::sort vs Google Highway VQSort
// vs Intel x86-simd-sort vs KDR short. Every contender's output is verified against std::sort.
//
// Build (see benchmarks/build_compare.sh for Highway):
//   AVX2   : g++ -std=c++17 -O3 -march=haswell          -Iinclude -I<highway> -I<x86-simd-sort>/src bench_compare.cpp libhwy_sort.a -pthread
//   AVX-512: g++ -std=c++17 -O3 -march=sapphirerapids -DXSS16 ... (same)   (-DXSS16 enables 16-bit x86-simd-sort)
// Run:  ./a.out 0   main table     ./a.out 1   std::string URLs     ./a.out 2   20M-element scaling rows
#include <kdr/kdr_sort.hpp>
#include "hwy/contrib/sort/vqsort.h"
#include "x86simdsort-static-incl.h"
#include <chrono>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#include <algorithm>
using namespace std; using clk=chrono::steady_clock;
#ifdef XSS16
constexpr size_t XSS_MIN=2;
#else
constexpr size_t XSS_MIN=4;
#endif
static double ms(clk::time_point a){ return chrono::duration<double,milli>(clk::now()-a).count(); }
template<class F> double timeit(const char* , F&& f, int reps){ vector<double> t; for(int i=0;i<reps;i++) t.push_back(f()); sort(t.begin(),t.end()); return t[t.size()/2]; }
template<class T> void row(const char* name, const vector<T>& base){
  size_t n=base.size(); vector<T> ref=base; double t_std;
  { auto a=clk::now(); sort(ref.begin(),ref.end()); t_std=ms(a); }
  int reps = t_std>150?3:(t_std>30?5:9);
  vector<T> v;
  double s=timeit("std",[&]{ v=base; auto a=clk::now(); sort(v.begin(),v.end()); return ms(a);},reps);
  { vector<T> w(base.begin(),base.begin()+min<size_t>(n,1000000)); kdr::sort(w); } // warm TLS scratch
  bool ok=true;
  double k=timeit("kdr",[&]{ v=base; auto a=clk::now(); kdr::sort(v); double t=ms(a); if(v!=ref) ok=false; return t;},reps);
  double q=-1; if constexpr(sizeof(T)>=2){ q=timeit("vq",[&]{ v=base; auto a=clk::now(); hwy::VQSort(v.data(),n,hwy::SortAscending()); double t=ms(a); if(v!=ref) ok=false; return t;},reps); }
  double x=-1; if constexpr(sizeof(T)>=XSS_MIN){ x=timeit("xss",[&]{ v=base; auto a=clk::now(); x86simdsortStatic::qsort(v.data(),n); double t=ms(a); if(v!=ref) ok=false; return t;},reps); }
  printf("%-26s N=%-9zu std %8.2f | VQSort %8s | x86-simd-sort %8s | KDR %8.2f | KDR/std x%.1f | %s\n",name,n,s,
    q<0?"n/a":to_string(q).substr(0,q<100?5:6).c_str(), x<0?"n/a":to_string(x).substr(0,x<100?5:6).c_str(), k, s/k, ok?"verified":"MISMATCH");
  fflush(stdout);
}
static mt19937_64 R(2026);
template<class T> vector<T> uni(size_t n){ vector<T> v(n); for(auto&x:v) x=(T)R(); return v; }
template<class T> vector<T> small(size_t n,uint64_t m){ vector<T> v(n); for(auto&x:v) x=(T)(R()%(m+1)); return v; }
template<class T> vector<T> fl(size_t n){ uniform_real_distribution<T> d(-1e9,1e9); vector<T> v(n); for(auto&x:v) x=d(R); return v; }
void strings(){
  size_t n=1000000; vector<string> base(n);
  const char* hosts[]={"https://www.example-store.com","https://api.example-store.com","https://cdn.example-media.org","https://shop.example-retail.net"};
  for(auto&s:base){ s=string(hosts[R()%4])+"/catalog/category-"+to_string(R()%200)+"/product/"+to_string(R()%1000000)+"?ref=campaign"+to_string(R()%50)+"&session="+to_string(R()); }
  vector<string> ref=base; auto a=clk::now(); sort(ref.begin(),ref.end()); double t0=ms(a);
  vector<string> v; vector<double> ts,tk; bool ok=true;
  for(int i=0;i<3;i++){ v=base; a=clk::now(); sort(v.begin(),v.end()); ts.push_back(ms(a)); v=base; a=clk::now(); kdr::sort(v); tk.push_back(ms(a)); if(v!=ref) ok=false; }
  sort(ts.begin(),ts.end()); sort(tk.begin(),tk.end());
  printf("%-26s N=%-9zu std %8.2f | KDR %8.2f | KDR/std x%.2f | %s (synthetic URLs)\n","std::string URLs",n,ts[1],tk[1],ts[1]/tk[1],ok?"verified":"MISMATCH"); fflush(stdout); (void)t0;
}
int main(int argc,char**argv){ int part=argc>1?atoi(argv[1]):0;
  if(part==0){
  row("u8 uniform",uni<uint8_t>(10000000)); row("i8 uniform",uni<int8_t>(10000000));
  row("u16 uniform",uni<uint16_t>(5000000)); row("i16 uniform",uni<int16_t>(5000000));
  row("u32 uniform",uni<uint32_t>(5000000)); row("u32 small range [0..10k]",small<uint32_t>(5000000,10000));
  { auto v=uni<uint32_t>(5000000); sort(v.begin(),v.end()); row("u32 pre-sorted",v);} 
  row("i32 uniform",uni<int32_t>(1000000)); row("u64 uniform",uni<uint64_t>(2000000));
  row("float uniform",fl<float>(2000000)); row("double uniform",fl<double>(2000000)); }
  if(part==1) strings();
  if(part==2){ for(size_t n: {1000000ul,5000000ul,20000000ul}){ char b[64]; snprintf(b,64,"u32 uniform %zuM",n/1000000); row(b,uni<uint32_t>(n)); }
               for(size_t n: {1000000ul,5000000ul,20000000ul}){ char b[64]; snprintf(b,64,"u64 uniform %zuM",n/1000000); row(b,uni<uint64_t>(n)); } }
}
