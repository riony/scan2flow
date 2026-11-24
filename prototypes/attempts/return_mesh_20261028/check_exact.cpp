#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/boost/graph/helpers.h>
#include <fstream>
#include <iostream>
using K=CGAL::Exact_predicates_exact_constructions_kernel;using Mesh=CGAL::Surface_mesh<K::Point_3>;
int main(int argc,char**argv){Mesh m;std::ifstream in(argv[1]);if(!CGAL::IO::read_OFF(in,m)){std::cerr<<"read failed: invalid polygon mesh\n";return 3;}std::vector<std::pair<Mesh::Face_index,Mesh::Face_index>> pairs;CGAL::Polygon_mesh_processing::self_intersections(faces(m),m,std::back_inserter(pairs));std::ofstream out(argv[2]);out<<"{\"faces\":"<<m.number_of_faces()<<",\"closed\":"<<(CGAL::is_closed(m)?"true":"false")<<",\"intersecting_pairs\":"<<pairs.size()<<"}\n";std::ofstream fp(std::string(argv[2])+".pairs");for(auto p:pairs)fp<<p.first.idx()<<" "<<p.second.idx()<<"\n";std::cerr<<"intersection pairs "<<pairs.size()<<"\n";return 0;}
