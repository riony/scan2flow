#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/intersections.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <queue>
#include <list>
#include <vector>
#include <stdexcept>
using K=CGAL::Exact_predicates_exact_constructions_kernel;
struct FaceInfo { int nesting=-1; };
using VB=CGAL::Triangulation_vertex_base_with_info_2<size_t,K>;
using FB=CGAL::Triangulation_face_base_with_info_2<FaceInfo,K,CGAL::Constrained_triangulation_face_base_2<K>>;
using TDS=CGAL::Triangulation_data_structure_2<VB,FB>;
using CDT=CGAL::Constrained_Delaunay_triangulation_2<K,TDS,CGAL::No_constraint_intersection_tag>;
using Pair=std::pair<size_t,size_t>;
void flood(CDT& cdt,CDT::Face_handle start,int nesting,std::list<CDT::Edge>& borders){
 std::queue<CDT::Face_handle> q;q.push(start);
 while(!q.empty()){
  auto f=q.front();q.pop();if(f->info().nesting!=-1)continue;f->info().nesting=nesting;
  for(int i=0;i<3;i++){auto g=f->neighbor(i);if(g->info().nesting!=-1)continue;
   CDT::Edge e(f,i);if(cdt.is_constrained(e))borders.push_back(e);else q.push(g);}
 }
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("expected input.txt output-prefix");
 std::ifstream in(argv[1]);size_t n,m;if(!(in>>n>>m)||n>500000||m>10000)throw std::runtime_error("invalid counts");
 std::map<size_t,K::Point_2> points;std::map<K::Point_2,size_t,K::Less_xy_2> unique;
 for(size_t i=0;i<n;i++){
  size_t id;double x,y;if(!(in>>id>>x>>y)||!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("bad point");
  K::Point_2 p(x,y);if(points.count(id))throw std::runtime_error("duplicate ID");
  if(unique.count(p))throw std::runtime_error("duplicate projected point IDs "+std::to_string(id)+" and "+std::to_string(unique.at(p)));
  points.emplace(id,p);unique.emplace(p,id);
 }
 std::vector<Pair> segments;std::set<Pair> unique_segments;
 for(size_t i=0;i<m;i++){
  size_t a,b;if(!(in>>a>>b)||a==b||!points.count(a)||!points.count(b))throw std::runtime_error("invalid segment");
  Pair e(std::min(a,b),std::max(a,b));if(!unique_segments.insert(e).second)throw std::runtime_error("duplicate constraint");segments.push_back(e);
 }
 for(size_t i=0;i<m;i++)for(size_t j=i+1;j<m;j++){
  auto [a,b]=segments[i];auto [c,d]=segments[j];K::Segment_2 s(points.at(a),points.at(b)),t(points.at(c),points.at(d));
  if(!CGAL::do_intersect(s,t))continue;
  bool shared=a==c||a==d||b==c||b==d;
  if(!shared)throw std::runtime_error("crossing constraints "+std::to_string(i)+" and "+std::to_string(j));
  if((c!=a&&c!=b&&s.has_on(points.at(c)))||(d!=a&&d!=b&&s.has_on(points.at(d)))||
     (a!=c&&a!=d&&t.has_on(points.at(a)))||(b!=c&&b!=d&&t.has_on(points.at(b))))
   throw std::runtime_error("overlapping constraints");
 }
 CDT cdt;std::map<size_t,CDT::Vertex_handle> handles;CDT::Face_handle hint;
 for(const auto& [id,p]:points){auto v=cdt.insert(p,hint);v->info()=id;handles[id]=v;hint=v->face();}
 // Atomic segments must not contain another vertex. Insertion must not split them.
 for(auto [a,b]:segments)cdt.insert_constraint(handles.at(a),handles.at(b));
 if(cdt.number_of_vertices()!=n||!cdt.is_valid())throw std::runtime_error("invalid CDT or added vertices");
 for(auto [a,b]:segments){CDT::Face_handle f;int i;
  if(!cdt.is_edge(handles.at(a),handles.at(b),f,i)||!cdt.is_constrained(CDT::Edge(f,i)))
   throw std::runtime_error("constraint not an atomic edge: "+std::to_string(a)+" "+std::to_string(b));
 }
 std::list<CDT::Edge> borders;flood(cdt,cdt.infinite_face(),0,borders);
 while(!borders.empty()){auto e=borders.front();borders.pop_front();auto g=e.first->neighbor(e.second);
  if(g->info().nesting==-1)flood(cdt,g,e.first->info().nesting+1,borders);
 }
 std::vector<std::array<size_t,3>> faces;std::set<size_t> used;
 for(auto f=cdt.finite_faces_begin();f!=cdt.finite_faces_end();++f){ // Complete convex domain: internal donor constraints are not holes.
  std::array<size_t,3> t{f->vertex(0)->info(),f->vertex(1)->info(),f->vertex(2)->info()};faces.push_back(t);used.insert(t.begin(),t.end());
 }
 if(used.size()!=n)throw std::runtime_error("input vertex omitted from marked domain");
 std::string prefix(argv[2]);std::ofstream out(prefix+".faces");out<<faces.size()<<"\n";
 for(auto t:faces)out<<t[0]<<" "<<t[1]<<" "<<t[2]<<"\n";
 std::ofstream edges(prefix+".constraints");for(auto [a,b]:segments)edges<<a<<" "<<b<<"\n";
 std::ofstream stats(prefix+".json");stats<<"{\"input_vertices\":"<<n<<",\"output_vertices\":"<<cdt.number_of_vertices()<<",\"constraints_verified\":"<<m<<",\"domain_mode\":\"all_finite_convex_hull\",\"domain_faces\":"<<faces.size()<<",\"all_input_ids_used\":true,\"CDT_valid\":true,\"added_vertices\":0}\n";
 std::cout<<"vertices "<<n<<" constraints "<<m<<" domain faces "<<faces.size()<<"\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}return 0;}
