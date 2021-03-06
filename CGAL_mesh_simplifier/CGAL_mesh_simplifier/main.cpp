#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/Euler_operations.h>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;

using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Plane_3 = Kernel::Plane_3;

using Mesh = CGAL::Surface_mesh<Point_3>;

using sm_vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using sm_halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
using sm_face_descriptor = boost::graph_traits<Mesh>::face_descriptor;

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using namespace std::literals::chrono_literals;


bool DeleteFacesByVertexIndices(Mesh& sm, const std::vector<int>& idx)
{
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
        auto vd = *(sm.vertices().first + i);
        // get halfedge_descriptor;
        auto hed = sm.halfedge(vd);
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


int main(int argc, char **argv)
{

    Mesh sm;
    std::filebuf fb;

    time_point<Clock> start = Clock::now();
    if (!fb.open ("data/Scan.off",std::ios::in))
    {
        return EXIT_FAILURE;
    }
    
    std::istream is(&fb);

    if (!CGAL::read_off(is, sm))
    {
        return EXIT_FAILURE;
    }

    time_point<Clock> end = Clock::now();
    milliseconds diff = duration_cast<milliseconds>(end - start);
    std::cout << "Read: " << diff.count() << "ms" << std::endl;

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

    // for demo purpose
    std::vector<int> idx(100000);
    std::generate(idx.begin(), idx.end(), [](){ static int i = 0; return i++; });

    start = Clock::now();

    DeleteFacesByVertexIndices(sm, idx);

    end = Clock::now();
    diff = duration_cast<milliseconds>(end - start);
    std::cout << "Process: " << diff.count() << "ms" << std::endl;

    printHelper("after remove face");

    std::ofstream of("simplified.off");
    if (!of.good())
    {
        return EXIT_FAILURE;
    }
    CGAL::write_off(of, sm);

    getchar();
    return EXIT_SUCCESS;
}