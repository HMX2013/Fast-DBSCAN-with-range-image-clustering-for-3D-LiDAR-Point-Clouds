#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>
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
#include <cmath>
#include <algorithm>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/version.hpp>

#include <dynamic_reconfigure/server.h>
#include <range_dbscan_segment/range_dbscan_segment_Config.h>
#include "cluster.h"
#include "segmentation_utility.h"
#define __APP_NAME__ "Range image segmentation"

#define UN_PROCESSED 0
#define PROCESSING 1
#define PROCESSED 2

inline bool comparePointClusters (const pcl::PointIndices &a, const pcl::PointIndices &b) {
    return (a.indices.size () < b.indices.size ());
}

typedef pcl::PointXYZ PointType;

// std::vector<cv::Scalar> _colors;
static bool _pose_estimation=false;
static const double _initial_quat_w = 1.0;

std::string output_frame_;
std::string non_ground_cloud_topic_;
std::string segmented_cloud_topic_;
std::string cluster_cloud_topic_;
std::string colored_cloud_topic_;
std::string cluster_cloud_trans_topic_;
std::string output_roi_topic_;
std_msgs::Header _velodyne_header;

// Pointcloud Filtering Parameters
float DETECT_MIN, DETECT_MAX;
float CLUSTER_THRESH, ClusterTolerance;
float segmentTh_H, segmentTh_V, d_th;

const float vehicle_len = 0.3;
const float vehicle_wid = 0.5;
const float detect_front = 6.0;
const float detect_side = 3.5;

int CLUSTER_MAX_SIZE, CLUSTER_MIN_SIZE, MinPts, MinClusterSize, MaxClusterSize;
float eps_coeff;

tf::TransformListener *_transform_listener;
tf::StampedTransform *_transform;


const int8_t neighbor_row = 1;
const int8_t neighbor_column = 10;

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
  ros::NodeHandle private_nh;

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

  dynamic_reconfigure::Server<range_dbscan_segment::range_dbscan_segment_Config> server;
  dynamic_reconfigure::Server<range_dbscan_segment::range_dbscan_segment_Config>::CallbackType f;

  pcl::PointCloud<pcl::PointXYZI>::Ptr colored_segmentation;

  ros::Subscriber sub_lidar_points;
  ros::Publisher pub_cloud_ground;
  ros::Publisher pub_cloud_clusters;
  ros::Publisher pub_jsk_bboxes;
  ros::Publisher pub_autoware_objects;
  ros::Publisher pub_clusters_transformed_;
  ros::Publisher publish_detected_objects_;
  ros::Publisher pub_colored_cluster_cloud_;
  ros::Publisher pub_roi_region_;

  void Roi_ProjectPointCloud(const sensor_msgs::PointCloud2::ConstPtr& laserRosCloudMsg, const float detection_min, const float detection_max);
  void range_image_search(int row, int col, std::vector<int> &k_indices, int &core_pt_ring);
  void RangeImageSegment(autoware_msgs::CloudClusterArray &in_out_clusters);
  void calculate_index2rc(int index, int &i, int &j);

  void Mainloop(const sensor_msgs::PointCloud2::ConstPtr& lidar_points);
  void publish_colored_cluster_cloud(const std_msgs::Header& header);
  void publish_autoware_cloudclusters(const ros::Publisher *in_publisher, const autoware_msgs::CloudClusterArray &in_clusters,
                          const std::string &in_target_frame, const std_msgs::Header &in_header);
  void publish_transformed_cloudclusters(const ros::Publisher *in_publisher, const autoware_msgs::CloudClusterArray &in_clusters,
                          const std::string &in_target_frame, const std_msgs::Header &in_header);
  void publish_transformed_clusters(const ros::Publisher *in_publisher, const autoware_msgs::CloudClusterArray &in_clusters,
                            const std::string &in_target_frame, const std_msgs::Header &in_header);

  void publish_detected_objects(const autoware_msgs::CloudClusterArray &in_clusters);
  void publish_roi_region(const std_msgs::Header& header);

 public:
  cloud_segmentation();
  ~cloud_segmentation() {};

  void allocateMemory(){
    laserCloudIn.reset(new pcl::PointCloud<PointType>());
    laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());
    colored_segmentation.reset(new pcl::PointCloud<pcl::PointXYZI>());
    fullCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);

    queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];

    allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

    std::pair<int8_t, int8_t> neighbor;

    // search area for the dbscan
    for (int8_t i = -neighbor_row; i <= neighbor_row; i++)
    {
      for (int8_t j = -neighbor_column; j <= neighbor_column; j++)
      {
        neighbor.first = i;
        neighbor.second = j;
        neighborIterator.push_back(neighbor);
      }
    }
  }

  void resetParameters(){
    laserCloudIn->clear();
            
    colored_segmentation->clear();
    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    labelCount = 1;
    cluster_indices.clear();
  }

};

// Dynamic parameter server callback function
void dynamicParamCallback(range_dbscan_segment::range_dbscan_segment_Config& config, uint32_t level)
{
  // Pointcloud Filtering Parameters

  DETECT_MIN = config.detect_min;
  DETECT_MAX = config.detect_max;

  MinPts=config.CorePointMinPt;
  eps_coeff=config.eps_coeff;
  ClusterTolerance=config.ClusterTolerance;
  MinClusterSize=config.MinClusterSize;
  MaxClusterSize=config.MaxClusterSize;
  segmentTh_H=config.segmentTh_H;
  segmentTh_V=config.segmentTh_V;
  d_th=config.d_th;
}


cloud_segmentation::cloud_segmentation():private_nh("~"){

  allocateMemory();

  /* Initialize tuning parameter */
  private_nh.param<std::string>("non_ground_cloud_topic", non_ground_cloud_topic_, "/non_ground_cloud_front");
  ROS_INFO("non_ground_cloud_topic: %s", non_ground_cloud_topic_.c_str());

  private_nh.param<std::string>("output_frame", output_frame_, "velodyne_1");
  ROS_INFO("output_frame: %s", output_frame_.c_str());

  private_nh.param<std::string>("colored_cloud_topic", colored_cloud_topic_, "/segmentation/colored_cloud");
  ROS_INFO("colored_cloud_topic: %s", colored_cloud_topic_.c_str());

  // private_nh.param<std::string>("segmented_cloud_topic", segmented_cloud_topic_, "/segmentation/segmented_cloud");
  // ROS_INFO("segmented_cloud_topic: %s", segmented_cloud_topic_.c_str());

  private_nh.param<std::string>("cluster_cloud_trans_topic", cluster_cloud_trans_topic_, "/segmentation/cluster_cloud_trans");
  ROS_INFO("cluster_cloud_trans_topic: %s", cluster_cloud_trans_topic_.c_str());

  private_nh.param<std::string>("output_roi_topic", output_roi_topic_, "/point_roi_front_");
  ROS_INFO("output_point_topic: %s", output_roi_topic_.c_str());

  sub_lidar_points = nh.subscribe(non_ground_cloud_topic_, 1, &cloud_segmentation::Mainloop, this);

  pub_colored_cluster_cloud_ = nh.advertise<sensor_msgs::PointCloud2> (colored_cloud_topic_, 1);
  pub_clusters_transformed_ = nh.advertise<autoware_msgs::CloudClusterArray>(cluster_cloud_trans_topic_, 1);
  publish_detected_objects_ = nh.advertise<autoware_msgs::DetectedObjectArray>("/segmentation/detected_objects", 1);
  pub_roi_region_ = nh.advertise<geometry_msgs::PolygonStamped>(output_roi_topic_, 2);

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


void cloud_segmentation::publish_colored_cluster_cloud(const std_msgs::Header& header)
{
  sensor_msgs::PointCloud2 colored_segmentation_ros;

  // extract segmented cloud for visualization
  if (pub_colored_cluster_cloud_.getNumSubscribers() != 0){
    pcl::toROSMsg(*colored_segmentation, colored_segmentation_ros);
    colored_segmentation_ros.header = header;
    pub_colored_cluster_cloud_.publish(colored_segmentation_ros);
  }
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

  //ROS_INFO("cloudSize is %d",cloudSize);

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
    // if (range < detection_min || range > detection_max || (horizonAngle > -180 && horizonAngle < -90) || thisPoint.z > 0.4)
    //   continue;

    // labelMat.at<int>(rowIdn, columnIdn) = -1;
    // rangeMat.at<float>(rowIdn, columnIdn) = range;
    // index = columnIdn  + rowIdn * Horizon_SCAN;
    // fullCloud->points[index] = thisPoint;


    // ROI
    if (((thisPoint.x > 0.0 && thisPoint.x < detect_front && thisPoint.y < detect_side && thisPoint.y > -(detect_side + vehicle_len)) ||
         (thisPoint.x > -vehicle_wid && thisPoint.x < 0.0 && thisPoint.y > 0.0 && thisPoint.y < detect_side)) &&
        thisPoint.z < 0.5)
    {
      labelMat.at<int>(rowIdn, columnIdn) = -1;
      rangeMat.at<float>(rowIdn, columnIdn) = range;
      index = columnIdn  + rowIdn * Horizon_SCAN;
      fullCloud->points[index] = thisPoint;
    }
    else 
    {continue;}

  }
}


void cloud_segmentation::range_image_search(int row, int col, std::vector<int> &k_indices, int &core_pt_ring){
  k_indices.clear();
  k_indices.push_back(col + row * Horizon_SCAN);
  core_pt_ring = 0;
  float dist, d_crtn, d_cur, d_criteria;
  float rou = 0.1;
  float L = 1.0;
  int fromIndX, fromIndY, thisIndX, thisIndY;

  fromIndX = row;
  fromIndY = col;
  int test=0;

  int ring[16] = {0};

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
    
    // prevent infinite loop (caused by put already examined point back)
    if (labelMat.at<int>(thisIndX, thisIndY) != -1)
      continue;

    int fromIndex_Pt = fromIndY + fromIndX * Horizon_SCAN;
    int thisIndex_Pt = thisIndY + thisIndX * Horizon_SCAN;

    float fromIndex_Pt_x = fullCloud->points[fromIndex_Pt].x;
    float fromIndex_Pt_y = fullCloud->points[fromIndex_Pt].y;
    float fromIndex_Pt_z = fullCloud->points[fromIndex_Pt].z;

    float thisIndex_Pt_x = fullCloud->points[thisIndex_Pt].x;
    float thisIndex_Pt_y = fullCloud->points[thisIndex_Pt].y;
    float thisIndex_Pt_z = fullCloud->points[thisIndex_Pt].z;

    dist = sqrt( (fromIndex_Pt_x-thisIndex_Pt_x)*(fromIndex_Pt_x-thisIndex_Pt_x)+
                  (fromIndex_Pt_y-thisIndex_Pt_y)*(fromIndex_Pt_y-thisIndex_Pt_y)+
                  (fromIndex_Pt_z-thisIndex_Pt_z)*(fromIndex_Pt_z-thisIndex_Pt_z) );

    d_criteria = eps_coeff * 0.0349 * sqrt(2) * rangeMat.at<float>(fromIndX, fromIndY);
    //ROS_INFO("d_criteria is %f",d_criteria); //sqrt(2) * tan(2 / 180 * M_PI) *   sqrt(2)*rangeMat.at<float>(fromIndX, fromIndY);

    if (dist < d_criteria){
      k_indices.push_back(thisIndY + thisIndX * Horizon_SCAN);
      ring[thisIndX]++;
    }
    // test++;
    //ROS_INFO("test=%d",test);
  }

  for (int i = 0; i < 16; i++)
  {
    if (ring[i] >= 3)
      core_pt_ring++;
  }

}


// segmentation process
void cloud_segmentation::RangeImageSegment(autoware_msgs::CloudClusterArray &in_out_clusters){

  std::vector<int> core_pt_indices;
  std::vector<bool> is_noise(N_SCAN*Horizon_SCAN, false);
  std::vector<int> types(N_SCAN*Horizon_SCAN, UN_PROCESSED);
  int neighborPts;
  int core_pt_ring;
  // int MinPts = 30;
  int min_pts_per_cluster_ = 30;
  int max_pts_per_cluster_ = 7000;

  for (size_t i = 0; i < N_SCAN; ++i){
    for (size_t j = 0; j < Horizon_SCAN; ++j){

      int index_ = j + i * Horizon_SCAN;

      if (types[index_] == PROCESSED) {
        continue;
      }

      if (labelMat.at<int>(i, j) == -1)
      {
        range_image_search(i, j, core_pt_indices, core_pt_ring);
        neighborPts = core_pt_indices.size();
      }
      else
      {continue;};

      if (neighborPts <= MinPts && core_pt_ring >= 2)
      {
        is_noise[index_] = true;
        continue;
      }

      std::vector<int> seed_queue;
      seed_queue.push_back(index_);
      types[index_] = PROCESSED;

      //ROS_INFO("neighborPts=%d",neighborPts);

      for (int k1 = 0; k1 < neighborPts; k1++)
      {
        if (core_pt_indices[k1] != index_)
        {
          seed_queue.push_back(core_pt_indices[k1]);
          types[core_pt_indices[k1]] = PROCESSING;
        }
      } // for every point near the chosen core point.

      int sq_idx = 1;
      int i1, j1;

      //ROS_INFO("seed_queue=%d",seed_queue.size());

      //core_pt_indices.clear();

      while (sq_idx < seed_queue.size()) {
        int cloud_index = seed_queue[sq_idx];
        if (is_noise[cloud_index] || types[cloud_index] == PROCESSED){
          types[cloud_index] = PROCESSED;
          sq_idx++;
          continue; // no need to check neighbors.
        }

        calculate_index2rc(cloud_index,i1,j1);

        range_image_search(i1, j1, core_pt_indices, core_pt_ring);
        neighborPts = core_pt_indices.size();
        // ROS_INFO("core_pt_ring=%d", core_pt_ring);

        if (neighborPts >= MinPts && core_pt_ring >= 2)
        {
          for (int k2 = 0; k2 < neighborPts; k2++)
          {
            if (types[core_pt_indices[k2]] == UN_PROCESSED)
            {
              seed_queue.push_back(core_pt_indices[k2]);
              types[core_pt_indices[k2]] = PROCESSING;
            }
          }
        }
        core_pt_indices.clear();

        types[cloud_index] = PROCESSED;
        sq_idx++;
      }

      //ROS_INFO("seed_queue size is %d",seed_queue.size());

      // If this queue is satisfactory, add to the clusters
      if (seed_queue.size() >= min_pts_per_cluster_ && seed_queue.size () <= max_pts_per_cluster_){
        pcl::PointIndices pclIndices;
        pclIndices.indices.resize(seed_queue.size());
        for (int j = 0; j < seed_queue.size(); ++j) {
            pclIndices.indices[j] = seed_queue[j];
        }
        // These two lines should not be needed: (can anyone confirm?) -FF
        std::sort (pclIndices.indices.begin (), pclIndices.indices.end ());
        pclIndices.indices.erase (std::unique (pclIndices.indices.begin (), pclIndices.indices.end ()), pclIndices.indices.end ());

        // r.header = _velodyne_header;
        cluster_indices.push_back (pclIndices);   // We could avoid a copy by working directly in the vector
      }
      else
      {
        PCL_DEBUG("[pcl::extractEuclideanClusters] This cluster has %zu points, which is not between %u and %u points, so it is not a final cluster\n",
                  seed_queue.size (), min_pts_per_cluster_, max_pts_per_cluster_);
      }
    }
  }
  
  // Sort the clusters based on their size (largest one first)
  std::sort (cluster_indices.rbegin (), cluster_indices.rend (), comparePointClusters);

  unsigned int k = 0;
  int intensity_mark = 1;
  float  verticalAngle;

  std::vector<ClusterPtr> segment_clusters;
  pcl::PointXYZI cluster_color;

  for (auto& getIndices : cluster_indices)
  {
    int ring[16] = {0};
    for (auto& index : getIndices.indices){
      cluster_color.x=fullCloud->points[index].x;
      cluster_color.y=fullCloud->points[index].y;
      cluster_color.z=fullCloud->points[index].z;
      colored_segmentation->push_back(cluster_color);
      colored_segmentation->points.back().intensity = intensity_mark;

      verticalAngle = atan2(cluster_color.z, sqrt(cluster_color.x * cluster_color.x + cluster_color.y * cluster_color.y)) * 180 / M_PI;
      int8_t ring_index = (verticalAngle + ang_bottom) / ang_res_y;
      if (ring_index >= 0 || ring_index <= 15)
        ring[ring_index]++;
      //ROS_INFO("ring_index=%d", ring_index);
    }


    // if (*std::max_element(ring, ring + 16) / double(getIndices.indices.size()) > 0.9)
    //   continue;

    // reject the cluster with ring less than 2
    int8_t cluster_ring = 0;
    for (int8_t i = 0; i < 16; i++)
    {
      if (ring[i] >= 3)
        cluster_ring++;
    }
    
    if (cluster_ring <= 2)
      continue;

    ClusterPtr cluster(new Cluster());
    cluster->SetCloud(fullCloud, getIndices.indices, _velodyne_header, k, 1, 1, 1, "", "false");
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

void cloud_segmentation::calculate_index2rc(int index, int &r, int &c){
  int j;
  for (int i = 0; i < N_SCAN; ++i)
  {
    j = index - i * Horizon_SCAN;
    if (j <= Horizon_SCAN){
        r=i;
        c=j;
        break;
    }
  }
}

void cloud_segmentation::publish_transformed_clusters(const ros::Publisher *in_publisher, const autoware_msgs::CloudClusterArray &in_clusters,
                          const std::string &in_target_frame, const std_msgs::Header &in_header)
{
  if (in_target_frame != in_header.frame_id)
  {
    autoware_msgs::CloudClusterArray clusters_transformed;
    clusters_transformed.header = in_header;
    clusters_transformed.header.frame_id = in_target_frame;

    //ROS_INFO("in_target_frame is %s",in_target_frame.c_str());

    for (auto i = in_clusters.clusters.begin(); i != in_clusters.clusters.end(); i++)
    {
      autoware_msgs::CloudCluster cluster_transformed;
      cluster_transformed.header = in_header;

      try
      {
        _transform_listener->lookupTransform(in_target_frame, _velodyne_header.frame_id, ros::Time(),
                                             *_transform);
        pcl_ros::transformPointCloud(in_target_frame, *_transform, i->cloud, cluster_transformed.cloud);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->min_point, in_header.frame_id,
                                            cluster_transformed.min_point);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->max_point, in_header.frame_id,
                                            cluster_transformed.max_point);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->avg_point, in_header.frame_id,
                                            cluster_transformed.avg_point);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->centroid_point, in_header.frame_id,
                                            cluster_transformed.centroid_point);
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
    publish_detected_objects(clusters_transformed);
  } 
  else
  {
    in_publisher->publish(in_clusters);
    publish_detected_objects(in_clusters);
  }
}


void cloud_segmentation::publish_detected_objects(const autoware_msgs::CloudClusterArray &in_clusters)
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
  publish_detected_objects_.publish(detected_objects);
}

void cloud_segmentation::Mainloop(const sensor_msgs::PointCloud2::ConstPtr& lidar_points)
{
  _velodyne_header = lidar_points->header;
  autoware_msgs::CloudClusterArray clusters_cloud;
  clusters_cloud.header = _velodyne_header;

  const auto start_time = std::chrono::steady_clock::now();

  //roi and label points for segmentation
  Roi_ProjectPointCloud(lidar_points, DETECT_MIN, DETECT_MAX);
  //ProjectPointCloud(lidar_points);

  RangeImageSegment(clusters_cloud);

  // Time the whole process
  const auto end_time = std::chrono::steady_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "range  image segmentation method took: " << elapsed_time.count() << " milliseconds" << std::endl;

  //visualization, use indensity to show different color for each cluster.
  publish_colored_cluster_cloud(_velodyne_header);

  //pub_clusters_transformed_.publish(clusters_cloud);

  publish_transformed_clusters(&pub_clusters_transformed_, clusters_cloud, output_frame_, _velodyne_header);

  // visualize the ROI region
  publish_roi_region(_velodyne_header);


  resetParameters();
}


void cloud_segmentation::publish_roi_region(const std_msgs::Header& header){

  geometry_msgs::PolygonStamped ROI_Polygon;

  geometry_msgs::Point32 point1, point2, point3, point4, point5, point6;

  point1.x= 0.0;  point1.y= -(detect_side + vehicle_len); point1.z= 0;
  point2.x= detect_front;  point2.y= -(detect_side + vehicle_len); point2.z= 0;
  point3.x= detect_front;  point3.y=  detect_side; point3.z= 0;
  point4.x=-vehicle_wid;  point4.y=  detect_side; point4.z= 0;
  point5.x=-vehicle_wid;  point5.y=0;     point5.z= 0;
  point6.x=0;     point6.y=0;  point6.z= 0;
  ROI_Polygon.polygon.points.push_back(point1);
  ROI_Polygon.polygon.points.push_back(point2);
  ROI_Polygon.polygon.points.push_back(point3);
  ROI_Polygon.polygon.points.push_back(point4);
  ROI_Polygon.polygon.points.push_back(point5);
  ROI_Polygon.polygon.points.push_back(point6);
  ROI_Polygon.header.frame_id = header.frame_id;

  pub_roi_region_.publish(ROI_Polygon);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "range_dbscan_segment_node");

  tf::StampedTransform transform;
  tf::TransformListener listener;

  _transform = &transform;
  _transform_listener = &listener;

  cloud_segmentation cloud_segmentation_node;

  ros::spin();

  return 0;
}