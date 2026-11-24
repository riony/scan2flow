#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/IO/OFF.h>
#include <CGAL/boost/graph/helpers.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = K::Point_3;
using Vec = K::Vector_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Mesh>;
using Traits = CGAL::AABB_traits_3<K, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;
namespace PMP = CGAL::Polygon_mesh_processing;

// Compute both float64 canonical quality and float32 STL-reader quality, returning their minimum.
static double tri_quality_both(const Point& p0, const Point& p1, const Point& p2) {
  auto calc = [](double x0, double y0, double z0,
                 double x1, double y1, double z1,
                 double x2, double y2, double z2) -> double {
    double e0x = x1 - x0, e0y = y1 - y0, e0z = z1 - z0;
    double e1x = x2 - x1, e1y = y2 - y1, e1z = z2 - z1;
    double e2x = x0 - x2, e2y = y0 - y2, e2z = z0 - z2;
    double l0 = std::sqrt(e0x * e0x + e0y * e0y + e0z * e0z);
    double l1 = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    double l2 = std::sqrt(e2x * e2x + e2y * e2y + e2z * e2z);
    double cx = -e0y * e2z + e0z * e2y;
    double cy = -e0z * e2x + e0x * e2z;
    double cz = -e0x * e2y + e0y * e2x;
    double area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    double prod = l0 * l1 * l2;
    double denom = 3.0 * std::sqrt(3.0) * prod * prod;
    if (denom <= 1e-36 || area <= 1e-24) return 0.0;
    return 64.0 * area * area * area / denom;
  };
  double q64 = calc(p0.x(), p0.y(), p0.z(), p1.x(), p1.y(), p1.z(), p2.x(), p2.y(), p2.z());
  double q32 = calc(static_cast<float>(p0.x()), static_cast<float>(p0.y()), static_cast<float>(p0.z()),
                    static_cast<float>(p1.x()), static_cast<float>(p1.y()), static_cast<float>(p1.z()),
                    static_cast<float>(p2.x()), static_cast<float>(p2.y()), static_cast<float>(p2.z()));
  return std::min(q64, q32);
}

static Vec unit_normal(const Point& p0, const Point& p1, const Point& p2) {
  Vec c = CGAL::cross_product(p1 - p0, p2 - p0);
  double n = std::sqrt(c.squared_length());
  return (n > 1e-20) ? (c / n) : Vec(0.0, 0.0, 0.0);
}

static void face_vertices(const Mesh& mesh, Mesh::Face_index f,
                          Mesh::Vertex_index& v0, Mesh::Vertex_index& v1, Mesh::Vertex_index& v2) {
  auto h = mesh.halfedge(f);
  v0 = mesh.source(h);
  v1 = mesh.target(h);
  v2 = mesh.target(mesh.next(h));
}

static double face_quality(const Mesh& mesh, Mesh::Face_index f) {
  Mesh::Vertex_index v0, v1, v2;
  face_vertices(mesh, f, v0, v1, v2);
  return tri_quality_both(mesh.point(v0), mesh.point(v1), mesh.point(v2));
}

static Vec face_normal(const Mesh& mesh, Mesh::Face_index f) {
  Mesh::Vertex_index v0, v1, v2;
  face_vertices(mesh, f, v0, v1, v2);
  return unit_normal(mesh.point(v0), mesh.point(v1), mesh.point(v2));
}

static bool check_local_unconstrained_creases(
    const Mesh& mesh,
    const Mesh::Property_map<Mesh::Edge_index, bool>& constraints,
    const std::vector<Mesh::Face_index>& faces,
    double cos_min) {
  for (auto f : faces) {
    if (!f.is_valid() || mesh.is_removed(f)) continue;
    Vec nf = face_normal(mesh, f);
    auto h0 = mesh.halfedge(f);
    for (auto h : CGAL::halfedges_around_face(h0, mesh)) {
      auto e = mesh.edge(h);
      if (constraints[e]) continue;
      auto fn = mesh.face(mesh.opposite(h));
      if (!fn.is_valid() || mesh.is_removed(fn)) continue;
      Vec nfn = face_normal(mesh, fn);
      if (nf * nfn < cos_min) return false;
    }
  }
  return true;
}

static int run_quality_edge_flips(
    Mesh& mesh,
    const Mesh::Property_map<Mesh::Edge_index, bool>& constraints,
    double q_trigger,
    double min_normal_dot) {
  int total_flips = 0;
  const double cos_27deg = std::cos(27.0 * M_PI / 180.0);
  for (int pass = 0; pass < 8; ++pass) {
    int flips = 0;
    for (auto e : mesh.edges()) {
      if (mesh.is_removed(e) || constraints[e]) continue;
      auto h = mesh.halfedge(e);
      auto g = mesh.opposite(h);
      auto f1 = mesh.face(h);
      auto f2 = mesh.face(g);
      if (!f1.is_valid() || !f2.is_valid() || f1 == f2) continue;
      double q1 = face_quality(mesh, f1);
      double q2 = face_quality(mesh, f2);
      double old_min = std::min(q1, q2);
      if (old_min > q_trigger) continue;

      auto u = mesh.source(h);
      auto v = mesh.target(h);
      auto w = mesh.target(mesh.next(h));
      auto z = mesh.target(mesh.next(g));
      if (w == z || u == v || w == u || w == v || z == u || z == v) continue;
      if (mesh.degree(u) < 4 || mesh.degree(v) < 4) continue;
      if (mesh.halfedge(w, z).is_valid()) continue;

      const Point& pu = mesh.point(u);
      const Point& pv = mesh.point(v);
      const Point& pw = mesh.point(w);
      const Point& pz = mesh.point(z);

      Vec n1 = unit_normal(pu, pv, pw);
      Vec n2 = unit_normal(pv, pu, pz);
      if (n1 * n2 < min_normal_dot) continue;

      Vec nn1 = unit_normal(pw, pu, pz);
      Vec nn2 = unit_normal(pz, pv, pw);
      if (n1 * nn1 < min_normal_dot || n2 * nn2 < min_normal_dot || nn1 * nn2 < min_normal_dot) continue;

      double nq1 = tri_quality_both(pw, pu, pz);
      double nq2 = tri_quality_both(pz, pv, pw);
      double new_min = std::min(nq1, nq2);
      if (new_min <= old_min + 1e-4) continue;

      bool crease_ok = true;
      std::pair<Mesh::Halfedge_index, Vec> outer[4] = {
          {mesh.next(h), nn2},
          {mesh.prev(h), nn1},
          {mesh.next(g), nn1},
          {mesh.prev(g), nn2}};
      for (auto& kv : outer) {
        auto oe = mesh.edge(kv.first);
        if (!constraints[oe]) {
          auto of = mesh.face(mesh.opposite(kv.first));
          if (of.is_valid() && of != f1 && of != f2) {
            if (face_normal(mesh, of) * kv.second < cos_27deg) {
              crease_ok = false;
              break;
            }
          }
        }
      }
      if (!crease_ok) continue;

      CGAL::Euler::flip_edge(h, mesh);
      ++flips;
      ++total_flips;
    }
    if (flips == 0) break;
  }
  return total_flips;
}

static int run_vertex_tangent_optimization(
    Mesh& mesh,
    const Mesh::Property_map<Mesh::Edge_index, bool>& constraints,
    const Mesh::Property_map<Mesh::Vertex_index, bool>& v_locked,
    const Tree& orig_tree,
    double q_trigger) {
  int moved = 0;
  const double cos_27deg = std::cos(27.0 * M_PI / 180.0);
  for (int pass = 0; pass < 5; ++pass) {
    int pass_moved = 0;
    std::set<Mesh::Vertex_index> cand_verts;
    for (auto f : mesh.faces()) {
      if (mesh.is_removed(f)) continue;
      if (face_quality(mesh, f) <= q_trigger) {
        Mesh::Vertex_index v0, v1, v2;
        face_vertices(mesh, f, v0, v1, v2);
        if (!v_locked[v0]) cand_verts.insert(v0);
        if (!v_locked[v1]) cand_verts.insert(v1);
        if (!v_locked[v2]) cand_verts.insert(v2);
      }
    }
    if (cand_verts.empty()) break;

    for (auto v : cand_verts) {
      if (mesh.is_removed(v) || v_locked[v]) continue;
      std::vector<Mesh::Face_index> inc_faces;
      std::vector<Vec> orig_normals;
      std::vector<Point> ring_pts;
      Vec avg_n(0.0, 0.0, 0.0);
      for (auto h : CGAL::halfedges_around_target(v, mesh)) {
        auto f = mesh.face(h);
        if (f.is_valid() && !mesh.is_removed(f)) {
          inc_faces.push_back(f);
          Vec nf = face_normal(mesh, f);
          orig_normals.push_back(nf);
          avg_n = avg_n + nf;
        }
        ring_pts.push_back(mesh.point(mesh.source(h)));
      }
      if (inc_faces.empty() || ring_pts.size() < 3) continue;
      double nlen = std::sqrt(avg_n.squared_length());
      if (nlen <= 1e-12) continue;
      avg_n = avg_n / nlen;

      double cur_min_q = 1e9;
      for (auto f : inc_faces) cur_min_q = std::min(cur_min_q, face_quality(mesh, f));
      if (cur_min_q > q_trigger) continue;

      Point p0 = mesh.point(v);
      Vec centroid_shift(0.0, 0.0, 0.0);
      double avg_edge = 0.0;
      for (const auto& rp : ring_pts) {
        Vec d = rp - p0;
        centroid_shift = centroid_shift + d;
        avg_edge += std::sqrt(d.squared_length());
      }
      centroid_shift = centroid_shift / static_cast<double>(ring_pts.size());
      avg_edge /= static_cast<double>(ring_pts.size());
      centroid_shift = centroid_shift - (centroid_shift * avg_n) * avg_n;

      Vec ref = (std::abs(avg_n.x()) < 0.9) ? Vec(1.0, 0.0, 0.0) : Vec(0.0, 1.0, 0.0);
      Vec t1 = CGAL::cross_product(avg_n, ref);
      t1 = t1 / std::sqrt(t1.squared_length());
      Vec t2 = CGAL::cross_product(avg_n, t1);

      Point best_p = p0;
      double best_q = cur_min_q;

      bool strictly_coplanar = true;
      for (const auto& nf : orig_normals) {
        if (nf * avg_n < 0.999999) {
          strictly_coplanar = false;
          break;
        }
      }

      std::vector<Point> candidates;
      for (double a : {0.15, 0.3, 0.5, 0.75, 1.0}) {
        candidates.push_back(p0 + a * centroid_shift);
      }
      for (double s : {0.05, 0.12, 0.25, 0.4}) {
        double r = s * avg_edge;
        for (int k = 0; k < 12; ++k) {
          double ang = k * (2.0 * M_PI / 12.0);
          Vec step = r * (std::cos(ang) * t1 + std::sin(ang) * t2);
          candidates.push_back(p0 + step);
          candidates.push_back(p0 + 0.5 * centroid_shift + step);
        }
      }

      for (Point cand : candidates) {
        Point proj = orig_tree.closest_point(cand);
        if (!strictly_coplanar || CGAL::squared_distance(cand, proj) > 1e-10) {
          cand = proj;
        }
        mesh.point(v) = cand;
        bool valid = true;
        double min_q = 1e9;
        for (size_t i = 0; i < inc_faces.size(); ++i) {
          Vec nn = face_normal(mesh, inc_faces[i]);
          if (nn * orig_normals[i] < 0.985) {
            valid = false;
            break;
          }
          double qf = face_quality(mesh, inc_faces[i]);
          min_q = std::min(min_q, qf);
          if (min_q <= best_q + 1e-4) {
            valid = false;
            break;
          }
        }
        if (valid && check_local_unconstrained_creases(mesh, constraints, inc_faces, cos_27deg)) {
          best_q = min_q;
          best_p = cand;
        }
      }
      mesh.point(v) = best_p;
      if (best_q > cur_min_q + 1e-4) {
        ++moved;
        ++pass_moved;
      }
    }
    if (pass_moved == 0) break;
  }
  return moved;
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("input.off feature_edges.txt edge_length output.off");
    Mesh mesh;
    std::ifstream in(argv[1]);
    if (!CGAL::IO::read_OFF(in, mesh) || !CGAL::is_triangle_mesh(mesh) || !CGAL::is_closed(mesh))
      throw std::runtime_error("invalid closed triangular carrier");

    Mesh orig_mesh = mesh;
    Tree orig_tree(faces(orig_mesh).first, faces(orig_mesh).second, orig_mesh);
    orig_tree.accelerate_distance_queries();

    auto constraints = mesh.add_property_map<Mesh::Edge_index, bool>("e:locked", false).first;
    std::set<std::pair<size_t, size_t>> wanted;
    std::ifstream edges(argv[2]);
    size_t a, b;
    while (edges >> a >> b) wanted.emplace(std::min(a, b), std::max(a, b));

    std::vector<Mesh::Edge_index> locked;
    for (auto e : mesh.edges()) {
      auto h = mesh.halfedge(e);
      size_t s = mesh.source(h).idx(), t = mesh.target(h).idx();
      if (wanted.count({std::min(s, t), std::max(s, t)})) {
        constraints[e] = true;
        locked.push_back(e);
      }
    }
    if (locked.size() != wanted.size() || wanted.empty())
      throw std::runtime_error("feature graph not reproduced");

    double length = std::stod(argv[3]);

    run_quality_edge_flips(mesh, constraints, 0.15, 0.999);

    PMP::split_long_edges(locked, length, mesh, CGAL::parameters::edge_is_constrained_map(constraints));
    if (mesh.number_of_faces() > 10000000) throw std::runtime_error("triangle cap");

    PMP::isotropic_remeshing(
        mesh.faces(), length, mesh,
        CGAL::parameters::number_of_iterations(3)
            .number_of_relaxation_steps(2)
            .edge_is_constrained_map(constraints)
            .protect_constraints(true)
            .relax_constraints(false));

    auto v_locked = mesh.add_property_map<Mesh::Vertex_index, bool>("v:locked", false).first;
    auto update_v_locked = [&]() {
      for (auto v : mesh.vertices()) v_locked[v] = false;
      for (auto e : mesh.edges()) {
        if (!mesh.is_removed(e) && constraints[e]) {
          auto h = mesh.halfedge(e);
          v_locked[mesh.source(h)] = true;
          v_locked[mesh.target(h)] = true;
        }
      }
    };
    update_v_locked();

    double min_q_initial = 1e9;
    int bad_initial = 0;
    for (auto f : mesh.faces()) {
      double q = face_quality(mesh, f);
      min_q_initial = std::min(min_q_initial, q);
      if (q <= 0.05) ++bad_initial;
    }

    int total_flips = 0, total_relaxes = 0, total_splits = 0, total_collapses = 0;

    for (int outer = 0; outer < 6; ++outer) {
      total_flips += run_quality_edge_flips(mesh, constraints, 0.12, 0.995);
      total_relaxes += run_vertex_tangent_optimization(mesh, constraints, v_locked, orig_tree, 0.10);
      total_flips += run_quality_edge_flips(mesh, constraints, 0.12, 0.990);

      std::vector<Mesh::Edge_index> to_split;
      std::set<Mesh::Edge_index> seen_edges;
      for (auto f : mesh.faces()) {
        if (mesh.is_removed(f)) continue;
        if (face_quality(mesh, f) <= 0.072) {
          Mesh::Edge_index best_e;
          double best_len2 = -1.0;
          for (auto h : CGAL::halfedges_around_face(mesh.halfedge(f), mesh)) {
            auto e = mesh.edge(h);
            if (!constraints[e]) {
              double l2 = CGAL::squared_distance(mesh.point(mesh.source(h)), mesh.point(mesh.target(h)));
              if (l2 > best_len2) {
                best_len2 = l2;
                best_e = e;
              }
            }
          }
          if (best_e.is_valid() && best_len2 > 1e-8 && !seen_edges.count(best_e)) {
            seen_edges.insert(best_e);
            to_split.push_back(best_e);
          }
        }
      }
      if (to_split.empty()) break;

      for (auto e : to_split) {
        if (mesh.is_removed(e) || constraints[e]) continue;
        auto h = mesh.halfedge(e);
        auto g = mesh.opposite(h);
        auto f1 = mesh.face(h);
        auto f2 = mesh.face(g);
        if (!f1.is_valid() || !f2.is_valid() || f1 == f2) continue;
        if (std::min(face_quality(mesh, f1), face_quality(mesh, f2)) > 0.072) continue;

        Point pu = mesh.point(mesh.source(h));
        Point pv = mesh.point(mesh.target(h));
        Point mid = CGAL::midpoint(pu, pv);
        auto h_new = CGAL::Euler::split_edge(h, mesh);
        auto vm = mesh.target(h_new);
        mesh.point(vm) = mid;
        v_locked[vm] = false;
        constraints[mesh.edge(h_new)] = false;
        constraints[mesh.edge(h)] = false;

        auto h_split1 = CGAL::Euler::split_face(h_new, mesh.next(h), mesh);
        constraints[mesh.edge(h_split1)] = false;
        auto g_opp = mesh.opposite(h);
        auto h_split2 = CGAL::Euler::split_face(g_opp, mesh.next(mesh.opposite(h_new)), mesh);
        constraints[mesh.edge(h_split2)] = false;
        ++total_splits;
      }
      update_v_locked();
      total_relaxes += run_vertex_tangent_optimization(mesh, constraints, v_locked, orig_tree, 0.10);
      total_flips += run_quality_edge_flips(mesh, constraints, 0.12, 0.990);

      for (auto f : mesh.faces()) {
        if (mesh.is_removed(f) || face_quality(mesh, f) > 0.068) continue;
        for (auto h : CGAL::halfedges_around_face(mesh.halfedge(f), mesh)) {
          auto e = mesh.edge(h);
          if (constraints[e]) continue;
          auto u = mesh.source(h);
          auto v = mesh.target(h);
          if (v_locked[u] && v_locked[v]) continue;
          if (!CGAL::Euler::does_satisfy_link_condition(e, mesh)) continue;
          Point target_pt = v_locked[v] ? mesh.point(v) : (v_locked[u] ? mesh.point(u) : orig_tree.closest_point(CGAL::midpoint(mesh.point(u), mesh.point(v))));
          Point old_pu = mesh.point(u), old_pv = mesh.point(v);
          std::vector<Mesh::Face_index> check_faces;
          std::vector<Vec> check_normals;
          for (auto hv : CGAL::halfedges_around_target(u, mesh)) {
            auto cf = mesh.face(hv);
            if (cf.is_valid() && cf != mesh.face(h) && cf != mesh.face(mesh.opposite(h))) {
              check_faces.push_back(cf);
              check_normals.push_back(face_normal(mesh, cf));
            }
          }
          for (auto hv : CGAL::halfedges_around_target(v, mesh)) {
            auto cf = mesh.face(hv);
            if (cf.is_valid() && cf != mesh.face(h) && cf != mesh.face(mesh.opposite(h))) {
              check_faces.push_back(cf);
              check_normals.push_back(face_normal(mesh, cf));
            }
          }
          mesh.point(u) = target_pt;
          mesh.point(v) = target_pt;
          bool ok = true;
          for (size_t i = 0; i < check_faces.size(); ++i) {
            if (face_normal(mesh, check_faces[i]) * check_normals[i] < 0.985 ||
                face_quality(mesh, check_faces[i]) <= 0.072) {
              ok = false;
              break;
            }
          }
          mesh.point(u) = old_pu;
          mesh.point(v) = old_pv;
          if (ok) {
            auto v_kept = CGAL::Euler::collapse_edge(e, mesh);
            mesh.point(v_kept) = target_pt;
            update_v_locked();
            ++total_collapses;
            break;
          }
        }
      }
    }

    // Enforce unconstrained dihedral angle < 27.5 deg (well below 30 deg sharp-edge gate)
    const double cos_27p5deg = std::cos(27.5 * M_PI / 180.0);
    int chamfer_ops = 0;
    for (int cpass = 0; cpass < 8; ++cpass) {
      int changed = 0;
      for (auto e : mesh.edges()) {
        if (mesh.is_removed(e) || constraints[e]) continue;
        auto h = mesh.halfedge(e);
        auto g = mesh.opposite(h);
        auto f1 = mesh.face(h), f2 = mesh.face(g);
        if (!f1.is_valid() || !f2.is_valid() || f1 == f2) continue;
        Vec n1 = face_normal(mesh, f1), n2 = face_normal(mesh, f2);
        double cur_dot = n1 * n2;
        if (cur_dot <= cos_27p5deg) {
          auto u = mesh.source(h), v = mesh.target(h);
          auto w = mesh.target(mesh.next(h)), z = mesh.target(mesh.next(g));
          // Option 1: flip edge across ridge
          if (w != z && !mesh.halfedge(w, z).is_valid() && mesh.degree(u) >= 4 && mesh.degree(v) >= 4) {
            const Point &pu = mesh.point(u), &pv = mesh.point(v), &pw = mesh.point(w), &pz = mesh.point(z);
            double nq1 = tri_quality_both(pw, pu, pz);
            double nq2 = tri_quality_both(pz, pv, pw);
            Vec nn1 = unit_normal(pw, pu, pz), nn2 = unit_normal(pz, pv, pw);
            if (std::min(nq1, nq2) > 0.065 && nn1 * nn2 > cos_27p5deg && nn1 * n1 > cos_27p5deg && nn2 * n2 > cos_27p5deg) {
              CGAL::Euler::flip_edge(h, mesh);
              ++chamfer_ops;
              ++changed;
              continue;
            }
          }
          // Option 2: normal-tilt relaxation on unconstrained vertices of {w, z, u, v}
          Vec bis = n1 + n2;
          double blen = std::sqrt(bis.squared_length());
          if (blen > 1e-12) bis = bis / blen;
          for (auto rv : {w, z, u, v}) {
            if (v_locked[rv]) continue;
            Point p0 = mesh.point(rv);
            std::vector<Mesh::Face_index> inc_f;
            Vec avg_shift(0.0, 0.0, 0.0);
            int cnt = 0;
            for (auto hv : CGAL::halfedges_around_target(rv, mesh)) {
              avg_shift = avg_shift + (mesh.point(mesh.source(hv)) - p0);
              ++cnt;
              if (mesh.face(hv).is_valid()) inc_f.push_back(mesh.face(hv));
            }
            if (cnt > 0) avg_shift = avg_shift / static_cast<double>(cnt);

            Point best_pt = p0;
            double best_crease_dot = face_normal(mesh, f1) * face_normal(mesh, f2);
            std::vector<Point> trials;
            for (double d : {-0.00025, -0.00015, -0.00008, -0.00004, 0.00004, 0.00008, 0.00015, 0.00025}) {
              trials.push_back(p0 + d * bis);
              trials.push_back(p0 + d * n1);
              trials.push_back(p0 + d * n2);
            }
            for (double a : {0.1, 0.2, 0.35}) {
              trials.push_back(p0 + a * avg_shift);
            }
            for (const Point& cand : trials) {
              // Ensure surface deviation from original mesh stays <= 0.45 mm
              if (CGAL::squared_distance(cand, orig_tree.closest_point(cand)) > 2.0e-7) continue;
              mesh.point(rv) = cand;
              double min_q = 1e9;
              for (auto cf : inc_f) min_q = std::min(min_q, face_quality(mesh, cf));
              if (min_q <= 0.068) continue;
              double new_dot = face_normal(mesh, f1) * face_normal(mesh, f2);
              if (new_dot > best_crease_dot + 1e-5) {
                // Verify no other unconstrained edge on inc_f got worse than new_dot
                bool all_ok = true;
                for (auto cf : inc_f) {
                  Vec ncf = face_normal(mesh, cf);
                  for (auto hc : CGAL::halfedges_around_face(mesh.halfedge(cf), mesh)) {
                    if (constraints[mesh.edge(hc)]) continue;
                    auto nf = mesh.face(mesh.opposite(hc));
                    if (nf.is_valid() && ncf * face_normal(mesh, nf) < std::min(cur_dot, cos_27p5deg)) {
                      all_ok = false;
                      break;
                    }
                  }
                  if (!all_ok) break;
                }
                if (all_ok) {
                  best_crease_dot = new_dot;
                  best_pt = cand;
                }
              }
            }
            mesh.point(rv) = best_pt;
            if (best_crease_dot > cur_dot + 1e-5) {
              cur_dot = best_crease_dot;
              ++chamfer_ops;
              ++changed;
              if (cur_dot > cos_27p5deg) break;
            }
          }
        }
      }
      if (changed == 0) break;
    }

    if (mesh.number_of_faces() > 10000000 || !CGAL::is_closed(mesh))
      throw std::runtime_error("output topology or count cap");

    mesh.collect_garbage();

    double min_q_final = 1e9;
    int bad_final = 0;
    for (auto f : mesh.faces()) {
      double q = face_quality(mesh, f);
      min_q_final = std::min(min_q_final, q);
      if (q <= 0.05) ++bad_final;
    }

    double max_unconstrained_deg = 0.0;
    for (auto e : mesh.edges()) {
      if (constraints[e]) continue;
      auto h = mesh.halfedge(e);
      auto f1 = mesh.face(h), f2 = mesh.face(mesh.opposite(h));
      if (f1.is_valid() && f2.is_valid()) {
        double d = std::max(-1.0, std::min(1.0, face_normal(mesh, f1) * face_normal(mesh, f2)));
        double deg = std::acos(d) * 180.0 / M_PI;
        max_unconstrained_deg = std::max(max_unconstrained_deg, deg);
      }
    }

    std::ofstream out(argv[4]);
    if (!CGAL::IO::write_OFF(out, mesh, CGAL::parameters::stream_precision(17)))
      throw std::runtime_error("write failed");

    std::ofstream graph(std::string(argv[4]) + ".features");
    size_t nlocked = 0;
    for (auto e : mesh.edges()) {
      if (constraints[e]) {
        auto h = mesh.halfedge(e);
        graph << mesh.source(h).idx() << " " << mesh.target(h).idx() << "\n";
        nlocked++;
      }
    }
    std::cout << "input_features " << wanted.size()
              << " output_features " << nlocked
              << " vertices " << mesh.number_of_vertices()
              << " faces " << mesh.number_of_faces()
              << " min_q_initial " << min_q_initial
              << " bad_initial " << bad_initial
              << " min_q_final " << min_q_final
              << " bad_final " << bad_final
              << " max_unconstrained_deg " << max_unconstrained_deg
              << " flips " << total_flips
              << " relaxes " << total_relaxes
              << " splits " << total_splits
              << " collapses " << total_collapses
              << " chamfer_ops " << chamfer_ops << "\n";
  } catch (std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
