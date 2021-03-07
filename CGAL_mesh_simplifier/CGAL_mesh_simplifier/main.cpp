#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/polygon_mesh_processing.h>


#include <iostream>
#include <string>
#include <fstream>

#include<cstdlib>

#include<ctime>

clock_t start, end;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;

using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Plane_3 = Kernel::Plane_3;

using Mesh = CGAL::Surface_mesh<Point_3>;

using sm_vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using sm_halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
using sm_face_descriptor = boost::graph_traits<Mesh>::face_descriptor;

bool DeleteFacesByVertexIndices(Mesh& sm, const std::vector<int>& idx)
{
    sm.collect_garbage();

    const auto num_vertices = sm.num_vertices();
    // loop through all given indices.
    for (auto i : idx)
    {
        // validity check;
        if (num_vertices <= i)
        {
            return false;
        }

        // get vertex_descriptor;
        //auto vd = *(sm.vertices().first + i);
        auto vd = sm_vertex_descriptor(i);
        //std::cout << vd << std::endl;
        if (vd == sm.null_vertex())
            continue;
        // get halfedge_descriptor;
        auto hed = sm.halfedge(vd);
        if (hed == sm.null_halfedge())
            continue;

        // get all halfedges around vertex;
        auto heds = sm.halfedges_around_target(hed);

        // try mark faces as deleted;
        for (auto it = heds.begin(); it != heds.end(); ++it)
        {
            auto fd = sm.face(*it);
            if (fd != Mesh::null_face())
            {
                //std::cout << "removing face #" << fd << std::endl;
                CGAL::remove_face(fd, sm);
            }
        }

        // mark current vertex as deleted;
        sm.remove_vertex(vd);
    }

    // update memory;
    sm.collect_garbage();

    return true;
}

bool IsStandardHole(sm_halfedge_descriptor h, Mesh & mesh, double max_hole_diam, int min_num_hole_edges)
{
    int num_hole_edges = 0;
    CGAL::Bbox_3 hole_bbox;
    for (sm_halfedge_descriptor hc : CGAL::halfedges_around_face(h, mesh))
    {
        if (hc == mesh.null_halfedge())
            continue;
        const auto& p = mesh.point(target(hc, mesh));
        hole_bbox += p.bbox();
        ++num_hole_edges;
        if (hole_bbox.xmax() - hole_bbox.xmin() > max_hole_diam) return false;
        if (hole_bbox.ymax() - hole_bbox.ymin() > max_hole_diam) return false;
        if (hole_bbox.zmax() - hole_bbox.zmin() > max_hole_diam) return false;
    }
    if (num_hole_edges < min_num_hole_edges) return false;
    return true;
}

Mesh CleanupMesh(const Mesh& in)
{
    Mesh out;
    
    for (auto it = in.vertices_begin(); it != in.vertices_end(); ++it)
    {
        out.add_vertex(in.point(*it));
    }

    for (auto it = in.faces_begin(); it != in.faces_end(); ++it)
    {
        auto vrange = in.vertices_around_face(in.halfedge(*it));
        auto vbegin = vrange.begin();
        auto vend = vrange.end();

        auto vd0 = *vbegin++;
        auto vd1 = *vbegin++;
        auto vd2 = *vbegin;
        // assertion for 3 vertices for 1 face..
        assert(vbegin == vend);
        out.add_face(vd0, vd1, vd2);
    }

    return out;
}

void ExtractCycles(Mesh mesh)
{
    mesh.collect_garbage();

    std::vector<sm_halfedge_descriptor> border_cycles;
    CGAL::Polygon_mesh_processing::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));
    std::ofstream out("points.txt");
    for (sm_halfedge_descriptor h : border_cycles)
    {
        // 存储圆内的点
        std::vector<Point_3> cycle;
        for (sm_halfedge_descriptor hc : CGAL::halfedges_around_face(h, mesh))
        {
            if (hc == mesh.null_halfedge())
                continue;
            auto vd = source(hc, mesh);
            if (vd == mesh.null_vertex())
                continue;
            auto pt = mesh.point(vd);
            cycle.push_back(pt);
            out << pt.x() << " " << pt.y() << " " << pt.z() << std::endl;
        }
    }
    out.close();
}

void ExtractCyclesBAK(Mesh mesh)
{
    std::vector<sm_halfedge_descriptor> border_cycles;
    CGAL::Polygon_mesh_processing::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));
    std::ofstream out("points.txt");
    for (sm_halfedge_descriptor h : border_cycles)
    {
        // 存储圆内的点
        std::vector<Point_3> cycle;
        for (sm_halfedge_descriptor hc : CGAL::halfedges_around_face(h, mesh))
        {
            if (hc == mesh.null_halfedge())
                continue;
            auto vd = sm_vertex_descriptor(source(hc, mesh));
            if (vd == mesh.null_vertex())
                continue;
            cycle.push_back(mesh.point(source(hc, mesh)));
            out << mesh.point(source(hc, mesh)).x() << " " << mesh.point(source(hc, mesh)).y() << " " << mesh.point(source(hc, mesh)).z() << std::endl;
        }
    }
    out.close();
}


int main(int argc, char **argv)
{

    Mesh sm;
    std::filebuf fb;

    if (!fb.open("data/mesh_test.off", std::ios::in))
    {
        return EXIT_FAILURE;
    }

    std::istream is(&fb);

    if (!CGAL::read_off(is, sm))
    {
        return EXIT_FAILURE;
    }

    auto printHelper = [&sm](const std::string &str) {
        std::cout << str << " -          number_of_vertices: " << sm.number_of_vertices() << std::endl;
        std::cout << str << " -             number_of_edges: " << sm.number_of_edges() << std::endl;
        std::cout << str << " -             number_of_faces: " << sm.number_of_faces() << std::endl;
        std::cout << str << " -         number_of_halfedges: " << sm.number_of_halfedges() << std::endl;
        std::cout << str << " -  number_of_removed_vertices: " << sm.number_of_removed_vertices() << std::endl;
        std::cout << str << " -     number_of_removed_edges: " << sm.number_of_removed_edges() << std::endl;
        std::cout << str << " -     number_of_removed_faces: " << sm.number_of_removed_faces() << std::endl;
        std::cout << str << " - number_of_removed_halfedges: " << sm.number_of_removed_halfedges() << std::endl << std::endl;
    };

    printHelper("before remove face");

    std::vector<int> indices;
    std::ifstream in_indices("data/indices.txt", std::ios::in);
    std::string s;
    while (std::getline(in_indices, s)) // 逐行读取数据并存于s中，直至数据全部读取
    {
        indices.push_back(std::atoi(s.c_str()));
    }

    // for demo purpose

    start = clock();
    DeleteFacesByVertexIndices(sm, indices);
    printHelper("after remove face");

    sm = CleanupMesh(sm);
    printHelper("after rebuild");


    ExtractCycles(sm);
    end = clock();
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Total time:" << endtime * 1000 << "ms" << std::endl;	//ms为单位


    

    std::ofstream of("simplified.off");
    if (!of.good())
    {
        return EXIT_FAILURE;
    }
    CGAL::write_off(of, sm);
    std::cout << "end!" << std::endl;
    return EXIT_SUCCESS;
}