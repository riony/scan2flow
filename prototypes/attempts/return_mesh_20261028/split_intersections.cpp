#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/boost/graph/helpers.h>
#include <fstream>
#include <iostream>
#include <memory>
using K=CGAL::Exact_predicates_exact_constructions_kernel;
using Mesh=CGAL::Surface_mesh<K::Point_3>;
namespace PMP=CGAL::Polygon_mesh_processing;
struct Visitor: PMP::Corefinement::Default_visitor<Mesh> {
 Mesh::Property_map<Mesh::Face_index,int> ids;
 std::shared_ptr<int> current;
 Visitor(Mesh::Property_map<Mesh::Face_index,int> p):ids(p),current(std::make_shared<int>(-1)){}
 void before_subface_creations(Mesh::Face_index f,const Mesh&){*current=ids[f];}
 void after_subface_created(Mesh::Face_index f,const Mesh&){ids[f]=*current;}
};
int main(int argc,char**argv){
 if(argc!=4)return 2;
 Mesh m;std::ifstream in(argv[1]);if(!CGAL::IO::read_OFF(in,m)){std::cerr<<"read failed\n";return 3;}
 auto ids=m.add_property_map<Mesh::Face_index,int>("f:source",-1).first;std::ifstream idin(argv[2]);for(auto f:m.faces()){idin>>ids[f];if(!idin)return 4;}
 std::cerr<<"input "<<m.number_of_faces()<<" closed "<<CGAL::is_closed(m)<<"\n";
 Visitor visitor(ids);auto marked=m.add_property_map<Mesh::Edge_index,bool>("e:intersection",false).first; PMP::experimental::autorefine(m,CGAL::parameters::visitor(visitor).edge_is_constrained_map(marked)); bool ok=true; m.collect_garbage(); std::ofstream edges(std::string(argv[3])+"_cut_edges.txt"); for(auto e:m.edges()) if(marked[e]) {auto h=m.halfedge(e);edges<<m.source(h).idx()<<" "<<m.target(h).idx()<<"\n";}
 bool closed=CGAL::is_closed(m);bool si=PMP::does_self_intersect(m);
 std::cerr<<"repair "<<ok<<" faces "<<m.number_of_faces()<<" closed "<<closed<<" intersects "<<si<<"\n";
 std::ofstream out(std::string(argv[3])+".off");CGAL::IO::write_OFF(out,m,CGAL::parameters::stream_precision(17));
 std::ofstream idout(std::string(argv[3])+"_face_ids.txt");for(auto f:m.faces())idout<<ids[f]<<"\n";
 std::ofstream stats(std::string(argv[3])+".json");stats<<"{\"repair_return\":"<<(ok?"true":"false")<<",\"closed\":"<<(closed?"true":"false")<<",\"self_intersects_exact\":"<<(si?"true":"false")<<",\"faces\":"<<m.number_of_faces()<<",\"vertices\":"<<m.number_of_vertices()<<"}\n";
 return ok&&closed&&!si?0:5;
}
