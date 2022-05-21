#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>
// #include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include "autoware_msgs/CloudClusterArray.h"
#include <autoware_msgs/DetectedObjectArray.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <Eigen/Dense>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/version.hpp>

// #if (CV_MAJOR_VERSION == 3)

// #include "gencolors.cpp"

// #else

// #include <opencv2/contrib/contrib.hpp>
// #include <autoware_msgs/DetectedObjectArray.h>

// #endif

#include <dynamic_reconfigure/server.h>
#include <range_image_segment/range_image_segment_Config.h>
#include "cluster.h"
#include "segmentation_utility.h"
// using namespace cv;

// #define USING_ANGLE 1

typedef pcl::PointXYZ PointType;

std::vector<cv::Scalar> _colors;
static bool _pose_estimation=false;
static const double _initial_quat_w = 1.0;

std::string output_frame_;
std::string lidar_points_topic_;
std_msgs::Header _velodyne_header;

// Pointcloud Filtering Parameters
bool USE_PCA_BOX;
bool USE_TRACKING;
float VOXEL_GRID_SIZE, DETECT_MIN, DETECT_MAX;
Eigen::Vector4f ROI_MAX_POINT, ROI_MIN_POINT;
float CLUSTER_THRESH, ClusterTolerance;
float segmentTh_H, segmentTh_V, d_th;

int CLUSTER_MAX_SIZE, CLUSTER_MIN_SIZE, CorePointMinPt, MinClusterSize, MaxClusterSize;

tf::TransformListener *_transform_listener;
tf::StampedTransform *_transform;

geometry_msgs::Point transformPoint(const geometry_msgs::Point& point, const tf::Transform& tf)
{
  tf::Point tf_point;
  tf::pointMsgToTF(point, tf_point);

  tf_point = tf * tf_point;

  geometry_msgs::Point ros_point;
  tf::pointTFToMsg(tf_point, ros_point);

  return ros_point;
}


class cloud_segmentation
{
 private:

  ros::NodeHandle nh;
  // tf2_ros::Buffer tf2_buffer;
  // tf2_ros::TransformListener tf2_listener;

  // tf::TransformListener *_transform_listener;
  // tf::StampedTransform *_transform;

  cv::Mat rangeMat; // range matrix for range image
  cv::Mat labelMat; // label matrix for segmentaiton marking

  pcl::PointCloud<PointType>::Ptr laserCloudIn;
  pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;
  PointType nanPoint; // fill in fullCloud at each iteration

  std_msgs::Header cloudHeader;
  pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix

  uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
  uint16_t *queueIndY;

  uint16_t *allPushedIndX; // array for tracking points of a segmented object
  uint16_t *allPushedIndY;

  int labelCount;

  std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

  std::vector<pcl::PointIndices> cluster_indices;
  // std::vector<pcl::PointCloud<PointType>::Ptr> Fast_RI_clusters;

  dynamic_reconfigure::Server<range_image_segment::range_image_segment_Config> server;
  dynamic_reconfigure::Server<range_image_segment::range_image_segment_Config>::CallbackType f;

  pcl::PointCloud<pcl::PointXYZI>::Ptr segmentedCloudColor;

  ros::Subscriber sub_lidar_points;
  ros::Publisher pub_cloud_ground;
  ros::Publisher pub_cloud_clusters;
  ros::Publisher pub_jsk_bboxes;
  ros::Publisher pub_autoware_objects;
  ros::Publisher _pub_autoware_clusters_message;
  ros::Publisher _pub_autoware_detected_objects;
  ros::Publisher _pub_roi_area;
  ros::Publisher _pubSegmentedCloudColor;

  void Roi_ProjectPointCloud(const sensor_msgs::PointCloud2::ConstPtr& laserRosCloudMsg, const float detection_min, const float detection_max);
  void labelComponents(int row, int col);
  void RangeImageSegment(autoware_msgs::CloudClusterArray &in_out_clusters);


  void lidarPointsCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar_points);
  pcl::PointCloud<PointType>::Ptr roi_rectangle_filter(const sensor_msgs::PointCloud2::ConstPtr& laserRosCloudMsg, const float filter_res, const Eigen::Vector4f& min_pt, const Eigen::Vector4f& max_pt);
  void publishSegmentedCloudsColor(const std_msgs::Header& header);
  void publish_ROI_area(const std_msgs::Header& header);
  void publish_autoware_cloudclusters(const ros::Publisher *in_publisher, const autoware_msgs::CloudClusterArray &in_clusters,
                          const std::string &in_target_frame, const std_msgs::Header &in_header);
  void publishAutowareDetectedObjects(const autoware_msgs::CloudClusterArray &in_clusters);

 public:
  cloud_segmentation();
  ~cloud_segmentation() {};

  void allocateMemory(){
    laserCloudIn.reset(new pcl::PointCloud<PointType>());
    laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());
    segmentedCloudColor.reset(new pcl::PointCloud<pcl::PointXYZI>());
    fullCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);

    queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];

    allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);
  }

  void resetParameters(){
    laserCloudIn->clear();
            
    segmentedCloudColor->clear();
    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    labelCount = 1;
    // Fast_RI_clusters.clear();
    cluster_indices.clear();
  }

};

// Dynamic parameter server callback function
void dynamicParamCallback(range_image_segment::range_image_segment_Config& config, uint32_t level)
{
  // Pointcloud Filtering Parameters
  VOXEL_GRID_SIZE = config.voxel_grid_size;

  DETECT_MIN = config.detect_min;
  DETECT_MAX = config.detect_max;

  CorePointMinPt=config.CorePointMinPt;
  ClusterTolerance=config.ClusterTolerance;
  MinClusterSize=config.MinClusterSize;
  MaxClusterSize=config.MaxClusterSize;
  segmentTh_H=config.segmentTh_H;
  segmentTh_V=config.segmentTh_V;
  d_th=config.d_th;

  ROI_MAX_POINT = Eigen::Vector4f(config.roi_max_x, config.roi_max_y, config.roi_max_z, 1);
  ROI_MIN_POINT = Eigen::Vector4f(config.roi_min_x, config.roi_min_y, config.roi_min_z, 1);
}


cloud_segmentation::cloud_segmentation(){

  ros::NodeHandle private_nh("~");

  allocateMemory();

  std::string cloud_ground_topic;
  std::string cloud_clusters_topic;
  std::string jsk_bboxes_topic;
  //std::string autoware_objects_topic;

  // #if (CV_MAJOR_VERSION == 3)
  //   generateColors(_colors, 255);
  // #else
  //   cv::generateColors(_colors, 255);
  // #endif

  private_nh.param<std::string>("lidar_points_topic_", lidar_points_topic_, "/points_no_ground");
  private_nh.param<std::string>("output_frame_", output_frame_, "velodyne");

  sub_lidar_points = nh.subscribe(lidar_points_topic_, 1, &cloud_segmentation::lidarPointsCallback, this);
  _pubSegmentedCloudColor = nh.advertise<sensor_msgs::PointCloud2> ("/detection/segmented_cloud_color_marker", 1);
  _pub_autoware_clusters_message = nh.advertise<autoware_msgs::CloudClusterArray>("/detection/lidar_detector/cloud_clusters", 1);
  _pub_autoware_detected_objects = nh.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);
  _pub_roi_area = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/detection/roi_area", 1);


  // Dynamic Parameter Server & Function
  f = boost::bind(&dynamicParamCallback, _1, _2);
  server.setCallback(f);

  // Create point processor

  nanPoint.x = std::numeric_limits<float>::quiet_NaN();
  nanPoint.y = std::numeric_limits<float>::quiet_NaN();
  nanPoint.z = std::numeric_limits<float>::quiet_NaN();
  // nanPoint.intensity = -1;

  resetParameters();
}


void cloud_segmentation::publishSegmentedCloudsColor(const std_msgs::Header& header)
{
  sensor_msgs::PointCloud2 segmentedCloudColor_ros;
  
  // extract segmented cloud for visualization
  if (_pubSegmentedCloudColor.getNumSubscribers() != 0){
    pcl::toROSMsg(*segmentedCloudColor, segmentedCloudColor_ros);
    segmentedCloudColor_ros.header = header;
    _pubSegmentedCloudColor.publish(segmentedCloudColor_ros);
  }
}


pcl::PointCloud<PointType>::Ptr cloud_segmentation::roi_rectangle_filter(const sensor_msgs::PointCloud2::ConstPtr& laserRosCloudMsg, const float filter_res, const Eigen::Vector4f& min_pt, const Eigen::Vector4f& max_pt)
{
  // // Create the filtering object: downsample the dataset using a leaf size
  pcl::PointCloud<PointType>::Ptr input_cloud(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(*laserRosCloudMsg, *input_cloud);
  pcl::PointCloud<PointType>::Ptr cloud_roi(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr input_cloud_filter(new pcl::PointCloud<PointType>);

  if (filter_res > 0)
  {
      // Create the filtering object: downsample the dataset using a leaf size
      pcl::VoxelGrid<PointType> vg;
      vg.setInputCloud(input_cloud);
      vg.setLeafSize(filter_res, filter_res, filter_res);
      vg.filter(*input_cloud_filter);

      // Cropping the ROI
      pcl::CropBox<PointType> roi_region(true);
      roi_region.setMin(min_pt);
      roi_region.setMax(max_pt);
      roi_region.setInputCloud(input_cloud_filter);
      roi_region.filter(*cloud_roi);
  }
  else
  {
      // Cropping the ROI
      pcl::CropBox<PointType> roi_region(true);
      roi_region.setMin(min_pt);
      roi_region.setMax(max_pt);
      roi_region.setInputCloud(input_cloud);
      roi_region.filter(*cloud_roi);
  }

  // Removing the car roof region
  std::vector<int> indices;
  pcl::CropBox<PointType> roof(true);

  roof.setMin(Eigen::Vector4f(-1.63, -0.6, -1.86, 1));
  roof.setMax(Eigen::Vector4f(0.97, 0.6, 0.19, 1));

  roof.setInputCloud(cloud_roi);
  roof.filter(indices);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  for (auto& point : indices)
    inliers->indices.push_back(point);

  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud(cloud_roi);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cloud_roi);

  // return input_cloud_filter;
  return cloud_roi;
}

void cloud_segmentation::publish_autoware_cloudclusters(const ros::Publisher *in_publisher, const autoware_msgs::CloudClusterArray &in_clusters,
                          const std::string &in_target_frame, const std_msgs::Header &in_header)
{
  if (in_target_frame != in_header.frame_id)
  {
    autoware_msgs::CloudClusterArray clusters_transformed;
    clusters_transformed.header = in_header;
    clusters_transformed.header.frame_id = in_target_frame;
    for (auto i = in_clusters.clusters.begin(); i != in_clusters.clusters.end(); i++)
    {
      autoware_msgs::CloudCluster cluster_transformed;
      cluster_transformed.header = in_header;
      try
      {
        _transform_listener->lookupTransform(in_target_frame, _velodyne_header.frame_id, ros::Time(), *_transform);

        pcl_ros::transformPointCloud(in_target_frame, *_transform, i->cloud, cluster_transformed.cloud);

        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->min_point, in_header.frame_id, cluster_transformed.min_point);
                                  
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->max_point, in_header.frame_id, cluster_transformed.max_point);

        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->avg_point, in_header.frame_id, cluster_transformed.avg_point);

        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->centroid_point, in_header.frame_id, cluster_transformed.centroid_point);
  
        cluster_transformed.dimensions = i->dimensions;
        cluster_transformed.eigen_values = i->eigen_values;
        cluster_transformed.eigen_vectors = i->eigen_vectors;

        cluster_transformed.convex_hull = i->convex_hull;
        cluster_transformed.bounding_box.pose.position = i->bounding_box.pose.position;
        if(_pose_estimation)
        {
          cluster_transformed.bounding_box.pose.orientation = i->bounding_box.pose.orientation;
        }
        else
        {
          cluster_transformed.bounding_box.pose.orientation.w = _initial_quat_w;
        }
        clusters_transformed.clusters.push_back(cluster_transformed);
      }
      catch (tf::TransformException &ex)
      {
        ROS_ERROR("publishCloudClusters: %s", ex.what());
      }
    }
    in_publisher->publish(clusters_transformed);
    publishAutowareDetectedObjects(clusters_transformed);
  }
  else
  {
    in_publisher->publish(in_clusters);
    publishAutowareDetectedObjects(in_clusters);
  }
}

void cloud_segmentation::publishAutowareDetectedObjects(const autoware_msgs::CloudClusterArray &in_clusters)
{
  autoware_msgs::DetectedObjectArray detected_objects;
  detected_objects.header = in_clusters.header;

  for (size_t i = 0; i < in_clusters.clusters.size(); i++)
  {
    autoware_msgs::DetectedObject detected_object;
    detected_object.header = in_clusters.header;
    detected_object.label = "unknown";
    detected_object.score = 1.;
    detected_object.space_frame = in_clusters.header.frame_id;
    detected_object.pose = in_clusters.clusters[i].bounding_box.pose;
    detected_object.dimensions = in_clusters.clusters[i].dimensions;
    detected_object.pointcloud = in_clusters.clusters[i].cloud;
    detected_object.convex_hull = in_clusters.clusters[i].convex_hull;
    detected_object.valid = true;

    detected_objects.objects.push_back(detected_object);
  }
  _pub_autoware_detected_objects.publish(detected_objects);
}


void cloud_segmentation::publish_ROI_area(const std_msgs::Header& header)
{
  // Construct Bounding Boxes from the clusters
  jsk_recognition_msgs::BoundingBoxArray jsk_bboxes;
  jsk_bboxes.header = header;

  jsk_recognition_msgs::BoundingBox DtectionArea_car;
  DtectionArea_car.pose.position.x=0;
  DtectionArea_car.pose.position.y=0; 
  DtectionArea_car.pose.position.z=0;
  // DtectionArea_car.dimensions.x=ROI_MAX_POINT[1]-ROI_MIN_POINT[1];
  // DtectionArea_car.dimensions.y=ROI_MAX_POINT[2]-ROI_MIN_POINT[2];
  // DtectionArea_car.dimensions.z=4;

  DtectionArea_car.dimensions.x=10;
  DtectionArea_car.dimensions.y=6;
  DtectionArea_car.dimensions.z=4;

  DtectionArea_car.header.frame_id="velodyne";

  jsk_bboxes.boxes.push_back(DtectionArea_car);

  jsk_recognition_msgs::BoundingBox car_remove;
  car_remove.pose.position.x=0;
  car_remove.pose.position.y=0; 
  car_remove.pose.position.z=0;
  car_remove.dimensions.x=0.97+1.63;
  car_remove.dimensions.y=1.2;
  car_remove.dimensions.z=0.19+1.86;
  car_remove.header.frame_id="velodyne";
  jsk_bboxes.boxes.push_back(car_remove);
  
  _pub_roi_area.publish(jsk_bboxes);
}

void cloud_segmentation::Roi_ProjectPointCloud(const sensor_msgs::PointCloud2::ConstPtr& laserRosCloudMsg, const float detection_min, const float detection_max){
  float verticalAngle, horizonAngle, range;
  size_t rowIdn, columnIdn, index, cloudSize; 
  PointType thisPoint;

  cloudHeader = laserRosCloudMsg->header;
  cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
  pcl::fromROSMsg(*laserRosCloudMsg, *laserCloudIn);

  // // Remove Nan points
  // std::vector<int> indices;
  // pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

  // have "ring" channel in the cloud
  if (useCloudRing == true){
    pcl::fromROSMsg(*laserRosCloudMsg, *laserCloudInRing);
    if (laserCloudInRing->is_dense == false){
      ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
      ros::shutdown();
    }
  }

  cloudSize = laserCloudIn->points.size();

  ROS_INFO("cloudSize is %d",cloudSize);

  for (size_t i = 0; i < cloudSize; ++i){
    thisPoint.x = laserCloudIn->points[i].x;
    thisPoint.y = laserCloudIn->points[i].y;
    thisPoint.z = laserCloudIn->points[i].z;

    // ROS_INFO("pxx is %f, pyy is %f, pzz is %f", thisPoint.x, thisPoint.y, thisPoint.z);

    //find the row and column index in the iamge for this point
    if (useCloudRing == true){
      rowIdn = laserCloudInRing->points[i].ring;
    }
    else{
      verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
      rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
    }

    if (rowIdn < 0 || rowIdn >= N_SCAN)
      continue;

    horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

    columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
    if (columnIdn >= Horizon_SCAN)
      columnIdn -= Horizon_SCAN;

    if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
      continue;

    range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);

    // ROI
    if (range < detection_min || range > detection_max || (horizonAngle > -180 && horizonAngle < -90) || thisPoint.z > 0.4)
      continue;

    labelMat.at<int>(rowIdn, columnIdn) = -1;
    rangeMat.at<float>(rowIdn, columnIdn) = range;

    index = columnIdn  + rowIdn * Horizon_SCAN;
    fullCloud->points[index] = thisPoint;
  }
}


void cloud_segmentation::labelComponents(int row, int col){
  // use std::queue std::vector std::deque will slow the program down greatly
  float d1, d2, alpha, angle, d, d_nn1, Dmax;
  int fromIndX, fromIndY, thisIndX, thisIndY;
  bool lineCountFlag[N_SCAN] = {false};
  int Nan_Search_Num;

  queueIndX[0] = row;
  queueIndY[0] = col;
  int queueSize = 1;
  int queueStartInd = 0;
  int queueEndInd = 1;

  allPushedIndX[0] = row;
  allPushedIndY[0] = col;
  int allPushedIndSize = 1;

  pcl::PointCloud<PointType>::Ptr segments_ith(new pcl::PointCloud<PointType>);
  pcl::PointIndices pclIndices;

  //standard BFS 
  while (queueSize > 0)
  {
    // Pop point
    fromIndX = queueIndX[queueStartInd];
    fromIndY = queueIndY[queueStartInd];
    --queueSize;
    ++queueStartInd;
    // Mark popped point, The initial value of labelCount is 1.
    labelMat.at<int>(fromIndX, fromIndY) = labelCount;

    // Loop through all the neighboring grids of popped grid, neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
    for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter)
    {
      bool same_cluster_criterion = false;
      int nan_interval = 1;

      // new index
      thisIndX = fromIndX + (*iter).first;
      thisIndY = fromIndY + (*iter).second;

      // index should be within the boundary
      if (thisIndX < 0 || thisIndX >= N_SCAN)
        continue;
      // at range image margin (left or right side)
      if (thisIndY < 0)
        thisIndY = Horizon_SCAN - 1;
      if (thisIndY >= Horizon_SCAN)
        thisIndY = 0;

      //process NaN points && prevent infinite loop (caused by put already examined point back)
      if ((*iter).first == 0){
        Nan_Search_Num = 10;
      }
      else{
        Nan_Search_Num = 3;
      }

      while (labelMat.at<int>(thisIndX, thisIndY) != -1 && nan_interval < Nan_Search_Num)
      {
        // new index
        thisIndX = thisIndX + (*iter).first;
        thisIndY = thisIndY + (*iter).second;

        if (thisIndX < 0 || thisIndX >= N_SCAN)
          break;
        // at range image margin (left or right side)
        if (thisIndY < 0)
          thisIndY = Horizon_SCAN - 1;
        if (thisIndY >= Horizon_SCAN)
          thisIndY = 0;
        nan_interval++;
        // ROS_INFO("i=%d",i);
      }

      if (thisIndX < 0 || thisIndX >= N_SCAN)
        continue;

      if (labelMat.at<int>(thisIndX, thisIndY) != -1)
        continue;

/*-------------------------------------------------------------*/
      // if ((*iter).first == 0){
      //   d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), rangeMat.at<float>(thisIndX, thisIndY));
      //   d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), rangeMat.at<float>(thisIndX, thisIndY));

      //   alpha = segmentAlphaX * nan_interval;
      //   angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

      //   d_nn1 = sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * cos(alpha));
      //   Dmax = d2 * sin(alpha)/sin(segmentTh_V*M_PI/180 -alpha) + 3*0.03;
        
      //   // if (d_nn1 > Dmax)
      //   //   ROS_INFO("1");
        
      //   if (d_nn1 > Dmax)
      //     continue;
        
      //   // if (d_nn1 < Dmax)
      //   //   same_cluster_criterion = true;
      //   // ROS_INFO("0");
      //   int Pc_index = fromIndY + fromIndX * Horizon_SCAN;
      //   int Pc_index_m1 = fromIndY - (*iter).second + fromIndX * Horizon_SCAN;

      //   int Pn_index = thisIndY + thisIndX * Horizon_SCAN;
      //   int Pn_index_p1 = thisIndY + (*iter).second + thisIndX * Horizon_SCAN;

      //   Eigen::Vector2d Pc(fullCloud->points[Pc_index].x, fullCloud->points[Pc_index].y);
      //   Eigen::Vector2d Pc_m1(fullCloud->points[Pc_index_m1].x, fullCloud->points[Pc_index_m1].y);

      //   Eigen::Vector2d Pn(fullCloud->points[Pn_index].x, fullCloud->points[Pn_index].y);
      //   Eigen::Vector2d Pn_p1(fullCloud->points[Pn_index_p1].x, fullCloud->points[Pn_index_p1].y);

      //   Eigen::Vector2d V1 = (Pc_m1 - Pc).normalized();
      //   Eigen::Vector2d V2 = (Pn_p1 - Pn).normalized();

      //   Eigen::Vector2d V_bisector = (V1 + V2).normalized();
      //   Eigen::Vector2d V_central = -(Pc + Pn) / 2.0;

      //   double criterion_h1 = V_bisector.dot(V_central);
      //   double criterion_h2 = acos(V1.dot(V2) / (V1.norm() * V2.norm())) * 180 / M_PI;

      //   if (criterion_h1 <= 0 || criterion_h2 >= segmentTh_H)
      //     same_cluster_criterion = true;
      // }
      // else
      // {
      //   d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), rangeMat.at<float>(thisIndX, thisIndY));
      //   d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), rangeMat.at<float>(thisIndX, thisIndY));

      //   alpha = segmentAlphaY * nan_interval;
      //   angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

      //   d_nn1 = sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * cos(alpha));
      //   Dmax = d2 * sin(alpha)/sin(segmentTh_V*M_PI/180 -alpha) + 3*0.03;

      //   if (angle > segmentTh_V / 180.0 * M_PI)
      //       same_cluster_criterion = true;

      //   // if (d_nn1 < Dmax)
      //   //   same_cluster_criterion = true;
      // }
/*-------------------------------------------------------------*/
      // if ((*iter).first == 0)
      // {
      //   alpha = nan_interval * segmentAlphaX;
      //   d_nn1 = sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * cos(alpha));
      //   Dmax = d2 * sin(alpha)/sin(segmentTh_H*M_PI/180 -alpha) + 3*0.02;
      // }
      // else
      // {
      //   alpha = nan_interval * segmentAlphaY;
      //   d_nn1 = sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * cos(alpha));
      //   Dmax = d2 * sin(alpha)/sin(segmentTh_V*M_PI/180 -alpha) + 3*0.02;
      // }
      // if (d_nn1 < Dmax)
      //   same_cluster_criterion = true;
/*-------------------------------------------------------------*/
      d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                    rangeMat.at<float>(thisIndX, thisIndY));
      d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                    rangeMat.at<float>(thisIndX, thisIndY));

      if ((*iter).first == 0)
        alpha = nan_interval * segmentAlphaX;
      else
        alpha = nan_interval * segmentAlphaY;
      
      angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

      if (angle > segmentTh_H / 180.0 * M_PI)
          same_cluster_criterion = true;
/*-------------------------------------------------------------*/

      if (same_cluster_criterion)
      {
        queueIndX[queueEndInd] = thisIndX;
        queueIndY[queueEndInd] = thisIndY;
        ++queueSize;
        ++queueEndInd;

        labelMat.at<int>(thisIndX, thisIndY) = labelCount;
        lineCountFlag[thisIndX] = true;

        segments_ith->push_back(fullCloud->points[thisIndY + thisIndX * Horizon_SCAN]);

        pclIndices.indices.push_back(thisIndY + thisIndX * Horizon_SCAN);

        allPushedIndX[allPushedIndSize] = thisIndX;
        allPushedIndY[allPushedIndSize] = thisIndY;
        ++allPushedIndSize;
      }
    }
  }

  // check if this segment is valid
  bool feasibleSegment = false;

  if (allPushedIndSize >= 10)
    feasibleSegment = true;
  else if (allPushedIndSize >= segmentValidPointNum){
    int lineCount = 0;
    for (size_t i = 0; i < N_SCAN; ++i)
      if (lineCountFlag[i] == true)
        ++lineCount;
    if (lineCount >= segmentValidLineNum)
      feasibleSegment = true;
  }

  // segment is valid, mark these points
  if (feasibleSegment == true){
    ++labelCount;
    // Fast_RI_clusters.push_back(segments_ith);
    cluster_indices.push_back(pclIndices);
  }
  else{
  // segment is invalid, mark these points
    for (size_t i = 0; i < allPushedIndSize; ++i){
      labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
    }
  }
}


void cloud_segmentation::RangeImageSegment(autoware_msgs::CloudClusterArray &in_out_clusters){
  // segmentation process
  for (size_t i = 0; i < N_SCAN; ++i)
    for (size_t j = 0; j < Horizon_SCAN; ++j)
      if (labelMat.at<int>(i, j) == -1)
        labelComponents(i, j);

  unsigned int k = 0;
  int intensity_mark = 1;
  
  std::vector<ClusterPtr> segment_clusters;
  pcl::PointXYZI cluster_color;

  for (auto& getIndices : cluster_indices)
  {
    for (auto& index : getIndices.indices){
      cluster_color.x = fullCloud->points[index].x;
      cluster_color.y = fullCloud->points[index].y;
      cluster_color.z = fullCloud->points[index].z;

      segmentedCloudColor->push_back(cluster_color);
      segmentedCloudColor->points.back().intensity = intensity_mark;
    }

    ClusterPtr cluster(new Cluster());
    // cluster->SetCloud(in_cloud_ptr, getIndices.indices, _velodyne_header, k, (int) _colors[k].val[0],
    //                   (int) _colors[k].val[1],
    //                   (int) _colors[k].val[2], "", _pose_estimation);
    cluster->SetCloud(fullCloud, getIndices.indices, _velodyne_header, k, 1, 1, 1, "", _pose_estimation);

    segment_clusters.push_back(cluster);
    intensity_mark++;
    k++;
  }

  for (unsigned int i = 0; i < segment_clusters.size(); i++)
  {
    if (segment_clusters[i]->IsValid())
    {
      autoware_msgs::CloudCluster cloud_cluster;
      segment_clusters[i]->ToROSMessage(_velodyne_header, cloud_cluster);
      in_out_clusters.clusters.push_back(cloud_cluster);
    }
  }
}


void cloud_segmentation::lidarPointsCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar_points)
{
  _velodyne_header = lidar_points->header;
  autoware_msgs::CloudClusterArray cloud_clusters;
  cloud_clusters.header = _velodyne_header;

  const auto start_time = std::chrono::steady_clock::now();

  //roi and label points for segmentation
  Roi_ProjectPointCloud(lidar_points, DETECT_MIN, DETECT_MAX);

  RangeImageSegment(cloud_clusters);

  // std::cout << "The size of Fast_RI_clusters is " << Fast_RI_clusters.size()  << std::endl;
  // std::cout << "The capacity of Fast_RI_clusters is " << Fast_RI_clusters.capacity()  << std::endl;
  // std::cout << "The Fast_RI_clusters[0] is " << Fast_RI_clusters[0]->points.size()  << std::endl;

  // Time the whole process
  const auto end_time = std::chrono::steady_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "range  image segmentation method took: " << elapsed_time.count() << " milliseconds" << std::endl;

  //visualization, use indensity to show different color for each cluster.
  //publish_ROI_area(_velodyne_header);

  publishSegmentedCloudsColor(_velodyne_header);

  //_pub_autoware_clusters_message.publish(cloud_clusters);

  publish_autoware_cloudclusters(&_pub_autoware_clusters_message, cloud_clusters, output_frame_, _velodyne_header);

  //ROS_INFO("The obstacle_detector_node found %d obstacles in %.3f second", int(prev_boxes_.size()), float(elapsed_time.count()/1000.0));

  resetParameters();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "range_image_segment");

  cloud_segmentation cloud_segmentation_node;

  ros::spin();

  return 0;
}