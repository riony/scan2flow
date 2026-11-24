#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/boost/graph/IO/OFF.h>
#include <CGAL/boost/graph/helpers.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <set>
#include <utility>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;
using Mesh=CGAL::Surface_mesh<K::Point_3>;
namespace PMP=CGAL::Polygon_mesh_processing;
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("input.off feature_edges.txt edge_length output.off");
 Mesh mesh;std::ifstream in(argv[1]);if(!CGAL::IO::read_OFF(in,mesh)||!CGAL::is_triangle_mesh(mesh)||!CGAL::is_closed(mesh))throw std::runtime_error("invalid closed triangular carrier");
 auto constraints=mesh.add_property_map<Mesh::Edge_index,bool>("e:locked",false).first;
 std::set<std::pair<size_t,size_t>> wanted;std::ifstream edges(argv[2]);size_t a,b;while(edges>>a>>b)wanted.emplace(std::min(a,b),std::max(a,b));
 std::vector<Mesh::Edge_index> locked;
 for(auto e:mesh.edges()){auto h=mesh.halfedge(e);size_t s=mesh.source(h).idx(),t=mesh.target(h).idx();if(wanted.count({std::min(s,t),std::max(s,t)})){constraints[e]=true;locked.push_back(e);}}
 if(locked.size()!=wanted.size()||wanted.empty())throw std::runtime_error("feature graph not reproduced");
 double length=std::stod(argv[3]);
 PMP::split_long_edges(locked,length,mesh,CGAL::parameters::edge_is_constrained_map(constraints));
 if(mesh.number_of_faces()>10000000)throw std::runtime_error("triangle cap");
 PMP::isotropic_remeshing(mesh.faces(),length,mesh,CGAL::parameters::number_of_iterations(3).number_of_relaxation_steps(1).edge_is_constrained_map(constraints).protect_constraints(true).relax_constraints(false));
 if(mesh.number_of_faces()>10000000||!CGAL::is_closed(mesh))throw std::runtime_error("output topology or count cap");
 mesh.collect_garbage();std::ofstream out(argv[4]);if(!CGAL::IO::write_OFF(out,mesh,CGAL::parameters::stream_precision(17)))throw std::runtime_error("write failed");
 std::ofstream graph(std::string(argv[4])+".features");size_t nlocked=0;
 for(auto e:mesh.edges())if(constraints[e]){auto h=mesh.halfedge(e);graph<<mesh.source(h).idx()<<" "<<mesh.target(h).idx()<<"\n";nlocked++;}
 std::cout<<"input_features "<<wanted.size()<<" output_features "<<nlocked<<" vertices "<<mesh.number_of_vertices()<<" faces "<<mesh.number_of_faces()<<"\n";
 }catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}return 0;}
