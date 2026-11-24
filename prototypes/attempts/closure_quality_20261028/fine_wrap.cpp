#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/OFF.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/boost/graph/helpers.h>
#include <fstream>
#include <iostream>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;using Mesh=CGAL::Surface_mesh<K::Point_3>;
namespace PMP=CGAL::Polygon_mesh_processing;
int main(int argc,char**argv){if(argc!=3)return 2;Mesh input,m;std::ifstream in(argv[1]);if(!CGAL::IO::read_OFF(in,input))return 3;
CGAL::alpha_wrap_3(input,0.008,0.0005,m);std::cerr<<"wrapped "<<m.number_of_faces()<<"\n";{std::ofstream out(std::string(argv[2])+"_raw.off");CGAL::IO::write_OFF(out,m,CGAL::parameters::stream_precision(17));}
PMP::remove_almost_degenerate_faces(m,CGAL::parameters::cap_threshold(-0.995).needle_threshold(20.0).collapse_length_threshold(0.001).flip_triangle_height_threshold(0.001));
 m.collect_garbage();std::cerr<<"remeshed "<<m.number_of_faces()<<"\n";std::ofstream out(std::string(argv[2])+".off");CGAL::IO::write_OFF(out,m,CGAL::parameters::stream_precision(17));std::ofstream js(std::string(argv[2])+".json");js<<"{\"faces\":"<<m.number_of_faces()<<",\"closed\":"<<(CGAL::is_closed(m)?"true":"false")<<",\"self_intersects\":"<<(PMP::does_self_intersect(m)?"true":"false")<<"}\n";}
