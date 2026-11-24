#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/intersections.h>
#include <CGAL/squared_distance_3.h>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <iostream>
using K=CGAL::Exact_predicates_exact_constructions_kernel;
int main(int argc,char**argv){
 if(argc!=4)return 2;
 std::ifstream in(argv[1]);std::string magic;size_t nv,nf,ne;in>>magic>>nv>>nf>>ne;if(magic!="OFF")return 3;
 std::vector<K::Point_3> p; p.reserve(nv);
 for(size_t i=0;i<nv;i++){double x,y,z;in>>x>>y>>z;p.emplace_back(x,y,z);}
 std::vector<std::array<size_t,3>> f(nf);for(auto& t:f){int n;in>>n>>t[0]>>t[1]>>t[2];if(n!=3)return 4;}
 if(!in)return 5;
 std::ifstream pairs(argv[2]);size_t a,b,id;size_t count=0,hit=0,shared=0,degenerate=0;std::vector<double> sep;
 while(pairs>>a>>b>>id){if(a>=nv||b>=nv||id>=nf)return 6;const auto&t=f[id];K::Segment_3 s(p[a],p[b]);K::Triangle_3 tri(p[t[0]],p[t[1]],p[t[2]]);count++;
 if(a==t[0]||a==t[1]||a==t[2]||b==t[0]||b==t[1]||b==t[2])shared++;
 if(s.is_degenerate()||tri.is_degenerate()){degenerate++;continue;}
 if(CGAL::do_intersect(s,tri))hit++;
 else {
 K::FT distance=std::min(CGAL::squared_distance(s.source(),tri),CGAL::squared_distance(s.target(),tri));
 for(int i=0;i<3;i++)distance=std::min(distance,CGAL::squared_distance(s,K::Segment_3(tri.vertex(i),tri.vertex((i+1)%3))));
 sep.push_back(std::sqrt(std::max(0.,CGAL::to_double(distance))));
 }
 }
 std::sort(sep.begin(),sep.end());std::ofstream out(argv[3]);out<<std::setprecision(17)<<"{\"reported\":"<<count<<",\"exact_segment_triangle_intersections\":"<<hit<<",\"shared_vertex_pairs\":"<<shared<<",\"degenerate_pairs\":"<<degenerate<<",\"exact_disjoint\":"<<sep.size();
 if(!sep.empty())out<<",\"separation_m\":{\"min\":"<<sep.front()<<",\"median\":"<<sep[sep.size()/2]<<",\"p90\":"<<sep[size_t(.9*(sep.size()-1))]<<",\"max\":"<<sep.back()<<"}";
 out<<"}\n";std::cout<<count<<" reported; "<<hit<<" exact intersections\n";
}
