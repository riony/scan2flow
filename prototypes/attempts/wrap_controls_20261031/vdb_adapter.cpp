#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/Morphology.h>
#include <tbb/global_control.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <stdexcept>
using V=openvdb::Vec3d;
struct Segment{V a,b;};
double dist2(V p,const Segment& s){V d=s.b-s.a;double den=d.lengthSqr();double t=den?std::clamp((p-s.a).dot(d)/den,0.0,1.0):0.;return (p-s.a-t*d).lengthSqr();}
double quality(const V&a,const V&b,const V&c){return (b-a).cross(c-a).length()/((b-a).lengthSqr()+(c-a).lengthSqr()+(c-b).lengthSqr()+1e-50);}
void dump(const openvdb::FloatGrid& g,const std::string& path){
 std::vector<openvdb::Vec3s> p;std::vector<openvdb::Vec3I> t;std::vector<openvdb::Vec4I> q;
 openvdb::tools::volumeToMesh(g,p,t,q,0.,0.,true);
 if(t.size()+2*q.size()>10000000)throw std::runtime_error("triangle cap");
 std::ofstream out(path);out<<std::setprecision(12)<<"OFF\n"<<p.size()<<" "<<t.size()+2*q.size()<<" 0\n";
 for(auto x:p)out<<x.x()<<" "<<x.y()<<" "<<x.z()<<"\n";
 for(auto x:t)out<<"3 "<<x.x()<<" "<<x.y()<<" "<<x.z()<<"\n";
 for(auto x:q){auto a=x[0],b=x[1],c=x[2],d=x[3];
  double ac=std::min(quality(V(p[a]),V(p[b]),V(p[c])),quality(V(p[a]),V(p[c]),V(p[d])));
  double bd=std::min(quality(V(p[a]),V(p[b]),V(p[d])),quality(V(p[b]),V(p[c]),V(p[d])));
  if(ac>=bd)out<<"3 "<<a<<" "<<b<<" "<<c<<"\n3 "<<a<<" "<<c<<" "<<d<<"\n";
  else out<<"3 "<<a<<" "<<b<<" "<<d<<"\n3 "<<b<<" "<<c<<" "<<d<<"\n";
 }
}
void check(const openvdb::FloatGrid& g){if(g.activeVoxelCount()>60000000)throw std::runtime_error("active voxel cap");}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("input.off segments.txt pitch closing_voxels mean_iterations output_prefix");
 tbb::global_control cap(tbb::global_control::max_allowed_parallelism,16);openvdb::initialize();
 std::ifstream in(argv[1]);std::string magic;size_t nv,nf,ne;in>>magic>>nv>>nf>>ne;if(magic!="OFF"||nv>10000000||nf>10000000)throw std::runtime_error("bad input");
 std::vector<openvdb::Vec3s>p(nv);std::vector<openvdb::Vec3I>t(nf);
 for(auto &x:p)in>>x.x()>>x.y()>>x.z();for(auto &x:t){int n;in>>n>>x.x()>>x.y()>>x.z();if(n!=3)throw std::runtime_error("non triangle");}
 std::ifstream sin(argv[2]);size_t ns;sin>>ns;std::vector<Segment>seg(ns);for(auto&s:seg)sin>>s.a.x()>>s.a.y()>>s.a.z()>>s.b.x()>>s.b.y()>>s.b.z();
 double pitch=std::stod(argv[3]);int closing=std::stoi(argv[4]),iterations=std::stoi(argv[5]);std::string out=argv[6];
 auto tr=openvdb::math::Transform::createLinearTransform(pitch);
 auto g=openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*tr,p,t,3.0f);check(*g);
 dump(*g,out+"_voxel.off");
 auto mask=openvdb::FloatGrid::create(1.0f);mask->setTransform(tr->copy());auto acc=mask->getAccessor();
 auto support=g->deepCopy();
 openvdb::tools::dilateActiveValues(support->tree(),4,openvdb::tools::NN_FACE_EDGE_VERTEX,openvdb::tools::EXPAND_TILES);
 support->tree().voxelizeActiveTiles();check(*support);
 for(auto it=support->cbeginValueOn();it;++it){V x=support->indexToWorld(it.getCoord());double md=1e30;for(auto&s:seg)md=std::min(md,dist2(x,s));double d=std::sqrt(md);double w=std::clamp((d-.013)/.013,0.,1.);if(w<1)acc.setValue(it.getCoord(),float(w));}
 support.reset();
 openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*g);
 if(closing){filter.offset(-closing*pitch,mask.get());check(*g);filter.offset(closing*pitch,mask.get());check(*g);}
 dump(*g,out+"_closed.off");
 for(int i=0;i<iterations;i++){filter.mean(1,mask.get());check(*g);}
 dump(*g,out+".off");
 std::ofstream stats(out+"_vdb.json");stats<<"{\"active_voxels\":"<<g->activeVoxelCount()<<",\"grid_memory_bytes\":"<<g->memUsage()<<",\"mask_memory_bytes\":"<<mask->memUsage()<<",\"feature_segments\":"<<ns<<",\"pitch\":"<<pitch<<",\"closing_voxels\":"<<closing<<",\"mean_iterations\":"<<iterations<<"}\n";
 }catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}return 0;}
