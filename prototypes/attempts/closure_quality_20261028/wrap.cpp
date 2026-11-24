#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/OFF.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/boost/graph/helpers.h>
#include <fstream>
#include <iostream>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;using Mesh=CGAL::Surface_mesh<K::Point_3>;
namespace PMP=CGAL::Polygon_mesh_processing;
int main(int argc,char**argv){if(argc!=3)return 2;Mesh input,m;std::ifstream in(argv[1]);if(!CGAL::IO::read_OFF(in,input))return 3;
CGAL::alpha_wrap_3(input,0.04,0.001,m);std::cerr<<"wrapped "<<m.number_of_faces()<<"\n";{std::ofstream out(std::string(argv[2])+"_raw.off");CGAL::IO::write_OFF(out,m,CGAL::parameters::stream_precision(17));}
PMP::isotropic_remeshing(faces(m),0.025,m,CGAL::parameters::number_of_iterations(5));std::cerr<<"remeshed "<<m.number_of_faces()<<"\n";std::ofstream out(std::string(argv[2])+".off");CGAL::IO::write_OFF(out,m,CGAL::parameters::stream_precision(17));std::ofstream js(std::string(argv[2])+".json");js<<"{\"faces\":"<<m.number_of_faces()<<",\"closed\":"<<(CGAL::is_closed(m)?"true":"false")<<",\"self_intersects\":"<<(PMP::does_self_intersect(m)?"true":"false")<<"}\n";}
