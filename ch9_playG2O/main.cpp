#include "Converter.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
struct point {
    double x, y, z;
    int id;
    int observedFrameID;
};
struct mapPoint {
    double x, y, z;
    int id;
};
const float fx = 535.4;
const float fy = 539.2;
const float cx = 320.1;
const float cy = 247.6;
const float mbf = 40.0;

int main(int argc, char **argv) {
    cout<<"function in. Hello World"<<endl;

    ///Read data
    ///Read observed points' XYZ, point ID and observed Frame Id.
    ifstream pointReader, mapPointReader, poseReader, planeReader;
    pointReader.open("../localBundlePoints.txt");
    string line;
    int count = 0;
    vector<point> points;
    while (getline(pointReader, line)) {
        stringstream ss(line);
        point p;
        if (count % 5 == 0)
            ss >> p.x;
        else if (count % 5 == 1)
            ss >> p.y;
        else if (count % 5 == 2)
            ss >> p.z;
        else if (count % 5 == 3)
            ss >> p.id;
        else if (count % 5 == 4)
            ss >> p.observedFrameID;
        if (count % 5 == 4) {
            //cout<<p.x<<" "<<p.y<<" "<<p.z<<" "<<p.id<<" "<<p.observedFrameID<<endl;
            points.push_back(p);
        }
        count++;
    }
    cout << "load " << points.size() << " observed points" << endl;

    ///Read Map Points XYZ and its point ID
    mapPointReader.open("../localBundleMapPoints.txt");
    string line2;
    int count2 = 0;
    vector<mapPoint> mapPoints;
    while (getline(mapPointReader, line2)) {
        stringstream ss(line2);
        mapPoint p;
        if (count2 % 4 == 0)
            ss >> p.x;
        else if (count2 % 4 == 1)
            ss >> p.y;
        else if (count2 % 4 == 2)
            ss >> p.z;
        else if (count2 % 4 == 3)
            ss >> p.id;
        if (count2 % 4 == 3) {
            //cout << p.x << " " << p.y << " " << p.z << " " << p.id << " " << endl;
            mapPoints.push_back(p);
        }
        count2++;
    }
    cout << "Load " << mapPoints.size() << " mappoints " << endl;

    ///Load camera poses at Frame 1,2 and 3
    cv::Mat pose2 = (cv::Mat_<float>(4, 4) << 0.99241, 0.0144781, -0.122115, 0.202981,
            -0.0136932, 0.99988, 0.00726406, -0.0262754,
            0.122205, -0.00553679, 0.992489, 0.0460141,
            0, 0, 0, 1);
    cv::Mat pose1 = (cv::Mat_<float>(4, 4) << 0.991981, 0.0239371, -0.124101, 0.201774,
            -0.0225897, 0.99967, 0.0122531, -0.0314431,
            0.124353, -0.00935147, 0.992194, 0.0487706,
            0, 0, 0, 1);
    cv::Mat pose0 = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);

//    //with FAKE data first
//    cv::Mat pose0 = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
//            0, 1, 0, 0,
//            0, 0, 1, 0,
//            0, 0, 0, 1);
//    cout << "defined pose0 " << endl<<pose0 << endl;
    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    cout << "define optimizer done " << endl;

////    ///Init the G2O parameters
////    g2o::SparseOptimizer myoptimizer;
////    g2o::BlockSolver_6_3::LinearSolverType *mylinearSolver;
////    mylinearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
////    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(mylinearSolver);
////
////    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
////    myoptimizer.setAlgorithm(solver);
////    myoptimizer.setVerbose(false);

    int maxKFid = 0;
    ///Load map poses
    for (int i = 0; i < 3; i++) {
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        if (i == 0)
            vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(pose0));
        if (i == 1)
            vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(pose1));
        if (i == 2)
            vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(pose2));
        vSE3->setId(i);
        if (i == 0)
            vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        cout << "add vertex " << i << " with pose " << endl << vSE3->estimate() << endl;
        if (i > maxKFid)
            maxKFid = i;
    }
    //cout << "store 3 pose vertex, current maxKFid is " << maxKFid << endl;

    ///Load map points
    int mapVertexPointNum = 0;
    for (int i = 0; i < mapPoints.size(); i++) {
        cv::Mat mapPointXYZ = (cv::Mat_<float>(3, 1) << mapPoints[i].x, mapPoints[i].y, mapPoints[i].z);
        Eigen::Matrix<double, 3, 1> pointXYZ;
        pointXYZ << mapPoints[i].x, mapPoints[i].y, mapPoints[i].z;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(ORB_SLAM2::Converter::toVector3d(mapPointXYZ));
        //vPoint->setEstimate(pointXYZ);
        int id = mapPoints[i].id + maxKFid + 1; ///G2O ID
        vPoint->setId(id);
        //Linearsolvertype 所以可以边缘化
        vPoint->setMarginalized(true);
        if (mapPoints[i].id < 10) { ///Just show first 10 mappoints
            cout << "add mappoint vertex " << id << " (point id " << mapPoints[i].id << ") with"
                 << vPoint->estimate().transpose() << endl;
        }
        optimizer.addVertex(vPoint);
        mapVertexPointNum++;
    }
    cout << "mapVertexNum " << mapVertexPointNum << endl;

    ///load observation edges
    int edgeNum = 0;
    for (int i = 0; i < points.size(); i++) {
        //cout << points[i].x << " " << points[i].y << " " << points[i].z << " " << points[i].id << " "<< points[i].observedFrameID << endl;
        Eigen::Matrix<double, 3, 1> obs;
        obs << points[i].x, points[i].y, points[i].z;
        g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(points[i].id + 1 + maxKFid)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(points[i].observedFrameID)));
        cout << "add edge with points vertex ID " << points[i].id + 1 + maxKFid << " and frame id "
             << points[i].observedFrameID << endl;
        e->setMeasurement(obs);
        cout << "measurement " << e->measurement().transpose() << endl;
        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity();
        e->setInformation(Info);
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//        e->setRobustKernel(rk);
//        rk->setDelta(thHuberStereo);
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        e->bf = mbf;
        optimizer.addEdge(e);

        if (points[i].id < 10) {///Just showed first 10 edges
            cout << "add new edge " << e->measurement().transpose() << " linked vertex0: "
                 <<points[i].id + 1 + maxKFid  << " vertex1: " << points[i].observedFrameID << endl;
        }
        edgeNum++;
    }
    cout << "store edge num " << edgeNum << endl;

    //*Step 9 开始优化
    optimizer.initializeOptimization();
    //先优化五次
    optimizer.optimize(5);
    cout<<"g2o done"<<endl;

//    ///Add Data for camera pose Vertex
//    int maxVertexID = 0;
//    //matrix 2 quat
//    Eigen::Matrix<double, 3, 3> R;
//    R << pose0.at<float>(0, 0), pose0.at<float>(0, 1), pose0.at<float>(0, 2),
//            pose0.at<float>(1, 0), pose0.at<float>(1, 1), pose0.at<float>(1, 2),
//            pose0.at<float>(2, 0), pose0.at<float>(2, 1), pose0.at<float>(2, 2);
//    Eigen::Matrix<double, 3, 1> t(pose0.at<float>(0, 3), pose0.at<float>(1, 3), pose0.at<float>(2, 3));
//    g2o::SE3Quat pose0Quat = g2o::SE3Quat(R, t);
//
//    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
//    vSE3->setEstimate(pose0Quat);
//    vSE3->setId(maxVertexID);
//    optimizer.addVertex(vSE3);
//    maxVertexID++;
//    std::cout << "pose vertex added, maxVertexId " << maxVertexID << endl;
//    ///Add data for mappoint vertex
//    int maxMapPointNum = 10;
//    for (int i = 0; i < maxMapPointNum; i++) {
//        Eigen::Matrix<double, 3, 1> location;
//        location << i, i, i;
//        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
//        vPoint->setEstimate(location);
//        vPoint->setId(i + maxVertexID);
//        vPoint->setMarginalized(true);
//        optimizer.addVertex(vPoint);
//    }
//    maxVertexID += maxMapPointNum;
//    std::cout << "mappoint vertex added, maxVertexId " << maxVertexID << endl;
//
//    ///Add observation edge
//    int maxEdgeNum = 10;
//    for (int i = 0; i < maxEdgeNum; i++) {
//        Eigen::Matrix<double, 3, 1> obs;
//        obs << i + 100, i + 100, i + 100;
//        g2o::EdgeStereoSE3ProjectXYZ *newEdge = new g2o::EdgeStereoSE3ProjectXYZ();
//        newEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(i + 1)));
//        newEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//        cout << "add edge with points vertex ID " << i + 1 << " and frame id "
//             << 0;
//        newEdge->setMeasurement(obs);
//        cout << "measurement " << newEdge->measurement().transpose() << endl;
//        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity();
//        newEdge->setInformation(Info);
//        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//        newEdge->fx = fx;
//        newEdge->fy = fy;
//        newEdge->cx = cx;
//        newEdge->cy = cy;
//        newEdge->bf = mbf;
//        optimizer.addEdge(newEdge);
//    }
//
//    optimizer.setVerbose(true);
//    //*Step 9 开始优化
//    optimizer.initializeOptimization();
//    //先优化五次
//    optimizer.optimize(5);
//    cout << "g2o done" << endl;

    return 0;
}