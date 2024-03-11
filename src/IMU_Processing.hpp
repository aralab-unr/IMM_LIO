#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"
#include <Eigen/Dense>

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset_imm();
  void Reset_imm(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset_ca();
  void Reset_ca(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset_ct();
  void Reset_ct(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  Eigen::Matrix<double, 12, 12> Q_imm;
  Eigen::Matrix<double, 12, 12> Q_ca;
  Eigen::Matrix<double, 12, 12> Q_ct;
  void Process_imm(const MeasureGroup &meas_imm,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, PointCloudXYZI::Ptr pcl_un_imm, state_ikfom &state_points_imm, Matrix<double, 23, 23>  &P_imm_prev);
  void Process_ca(const MeasureGroup &meas_ca,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ca, PointCloudXYZI::Ptr pcl_un_ca, state_ikfom &state_points_ca, Matrix<double, 23, 23>  &P_ca_prev);
  void Process_ct(const MeasureGroup &meas_ct,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, PointCloudXYZI::Ptr pcl_un_ct, state_ikfom &state_points_ct, Matrix<double, 23, 23>  &P_ct_prev);
  void get_in_acc(V3D &v);
  void get_in_angl(V3D &angl);
  
  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_imm;
  V3D cov_gyr_imm;
  V3D cov_acc_ca;
  V3D cov_gyr_ca;
  V3D cov_acc_ct;
  V3D cov_gyr_ct;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  Matrix<double, 23, 23> cov_imm;
  Matrix<double, 23, 23> cov_ca;
  Matrix<double, 23, 23> cov_ct;
  double first_lidar_time;
  double first_lidar_time_imm;
  double first_lidar_time_ca;
  double first_lidar_time_ct;

 private:
  void IMU_init_imm(const MeasureGroup &meas_imm, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, int &N_imm);
  void IMU_init_ca(const MeasureGroup &meas_ca, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ca, int &N_ca);
  void IMU_init_ct(const MeasureGroup &meas_ct, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, int &N_ct);
  void UndistortPcl_imm(const MeasureGroup &meas_imm, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, PointCloudXYZI &pcl_in_out_imm, state_ikfom &state_points_imm, Matrix<double, 23, 23>  &P_imm_prev);
  void UndistortPcl_ca(const MeasureGroup &meas_ca, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ca, PointCloudXYZI &pcl_in_out_ca, state_ikfom &state_points_ca, Matrix<double, 23, 23>  &P_ca_prev);
  void UndistortPcl_ct(const MeasureGroup &meas_ct, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, PointCloudXYZI &pcl_in_out_ct, state_ikfom &state_points_ct, Matrix<double, 23, 23>  &P_ct_prev);


  PointCloudXYZI::Ptr cur_pcl_un_;
  PointCloudXYZI::Ptr cur_pcl_un_imm;
  PointCloudXYZI::Ptr cur_pcl_un_ca;
  PointCloudXYZI::Ptr cur_pcl_un_ct;
  sensor_msgs::ImuConstPtr last_imu_;
  sensor_msgs::ImuConstPtr last_imu_imm;
  sensor_msgs::ImuConstPtr last_imu_ca;
  sensor_msgs::ImuConstPtr last_imu_ct;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<Pose6D> IMUpose_imm;
  vector<Pose6D> IMUpose_ca;
  vector<Pose6D> IMUpose_ct;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_acc_imm;
  V3D mean_acc_ca;
  V3D mean_acc_ct;
  V3D mean_gyr;
  V3D mean_gyr_imm;
  V3D mean_gyr_ca;
  V3D mean_gyr_ct;
  V3D angvel_last;
  V3D acc_s_last;
  V3D angvel_last_imm;
  V3D acc_s_last_imm;
  V3D angvel_last_ca;
  V3D acc_s_last_ca;
  V3D angvel_last_ct;
  V3D acc_s_last_ct;
  double start_timestamp_;
  double last_lidar_end_time_;
  double last_lidar_end_time_imm;
  double last_lidar_end_time_ca;
  double last_lidar_end_time_ct;
  int    init_iter_num = 1;
  int    init_iter_num_imm = 1;
  int    init_iter_num_ca = 1;
  int    init_iter_num_ct = 1;
  bool   b_first_frame_ = true;
  bool   b_first_frame_imm = true;
  bool   b_first_frame_ca = true;
  bool   b_first_frame_ct = true;
  bool   imu_need_init_ = true;
  bool   imu_need_init_imm = true;
  bool   imu_need_init_ca = true;
  bool   imu_need_init_ct = true;
  bool   P_init_imm = false;
  bool   P_init_ca = false;
  bool   P_init_ct = false;
  bool   P_init_imm_v2 = false;
  bool   P_init_ca_v2 = false;
  bool   P_init_ct_v2 = false;

  V3D in_acc;
  V3D in_angl;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  Q_imm = process_noise_cov();
  Q_ca = process_noise_cov();
  Q_ct = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  cov_acc_imm       = V3D(0.1, 0.1, 0.1);
  cov_gyr_imm       = V3D(0.1, 0.1, 0.1);
  mean_acc_imm      = V3D(0, 0, -1.0);
  mean_gyr_imm      = V3D(0, 0, 0);
  cov_acc_ca       = V3D(0.1, 0.1, 0.1);
  cov_gyr_ca      = V3D(0.1, 0.1, 0.1);
  mean_acc_ca      = V3D(0, 0, -1.0);
  mean_gyr_ca      = V3D(0, 0, 0);
  cov_acc_ct       = V3D(0.1, 0.1, 0.1);
  cov_gyr_ct       = V3D(0.1, 0.1, 0.1);
  mean_acc_ct      = V3D(0, 0, -1.0);
  mean_gyr_ct      = V3D(0, 0, 0);
  angvel_last         = Zero3d;
  angvel_last_imm     = Zero3d;
  angvel_last_ca     = Zero3d;
  angvel_last_ct      = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
  last_imu_imm.reset(new sensor_msgs::Imu());
  last_imu_ca.reset(new sensor_msgs::Imu());
  last_imu_ct.reset(new sensor_msgs::Imu());

  // in_acc = V3D(-1.0,-1.0,-1.0);
}


ImuProcess::~ImuProcess() {}

void ImuProcess::get_in_acc(V3D &v)
{
  v = in_acc;
}

void ImuProcess::get_in_angl(V3D &angl)
{
  angl = in_angl;
}

void ImuProcess::Reset_imm() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_imm    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  IMUpose_imm.clear();
  last_imu_imm.reset(new sensor_msgs::Imu());
  cur_pcl_un_imm.reset(new PointCloudXYZI());
}

void ImuProcess::Reset_ca() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_ca    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  IMUpose_ca.clear();
  last_imu_ca.reset(new sensor_msgs::Imu());
  cur_pcl_un_ca.reset(new PointCloudXYZI());
}

void ImuProcess::Reset_ct() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_ct    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  IMUpose_ct.clear();
  last_imu_ct.reset(new sensor_msgs::Imu());
  cur_pcl_un_ct.reset(new PointCloudXYZI());
}


void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}



// void ImuProcess::UndistortPcl_imm(const MeasureGroup &meas_imm, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, PointCloudXYZI &pcl_out_imm)
void ImuProcess::UndistortPcl_imm(const MeasureGroup &meas_imm, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, PointCloudXYZI &pcl_out_imm, state_ikfom &state_points_imm, Matrix<double, 23, 23>  &P_imm_prev)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu_imm = meas_imm.imu;
  v_imu_imm.push_front(last_imu_imm);
  const double &imu_beg_time_imm = v_imu_imm.front()->header.stamp.toSec();
  const double &imu_end_time_imm = v_imu_imm.back()->header.stamp.toSec();
  const double &pcl_beg_time_imm = meas_imm.lidar_beg_time;
  const double &pcl_end_time_imm = meas_imm.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out_imm = *(meas_imm.lidar);
  sort(pcl_out_imm.points.begin(), pcl_out_imm.points.end(), time_list);

  /*** Initialize IMU pose ***/
  state_ikfom imu_state_imm = state_points_imm;

  IMUpose_imm.clear();
  IMUpose_imm.push_back(set_pose6d(0.0, acc_s_last_imm, angvel_last_imm, imu_state_imm.vel, imu_state_imm.pos, imu_state_imm.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr_imm, acc_avr_imm, acc_imu_imm, vel_imu_imm, pos_imu_imm;
  M3D R_imu_imm;

  double dt_imm = 0;

  input_ikfom in_imm;

  for (auto it_imu_imm = v_imu_imm.begin(); it_imu_imm < (v_imu_imm.end() - 1); it_imu_imm++)
  {
    auto &&head_imm = *(it_imu_imm);
    auto &&tail_imm = *(it_imu_imm + 1);
    
    if (tail_imm->header.stamp.toSec() < last_lidar_end_time_imm)    continue;
    
    angvel_avr_imm <<0.5 * (head_imm->angular_velocity.x + tail_imm->angular_velocity.x),
                     0.5 * (head_imm->angular_velocity.y + tail_imm->angular_velocity.y),
                     0.5 * (head_imm->angular_velocity.z + tail_imm->angular_velocity.z);
    acc_avr_imm   <<0.5 * (head_imm->linear_acceleration.x + tail_imm->linear_acceleration.x),
                    0.5 * (head_imm->linear_acceleration.y + tail_imm->linear_acceleration.y),
                    0.5 * (head_imm->linear_acceleration.z + tail_imm->linear_acceleration.z);

    acc_avr_imm     = acc_avr_imm * G_m_s2 / mean_acc_imm.norm(); // - state_inout.ba;

    if(head_imm->header.stamp.toSec() < last_lidar_end_time_imm)
    {
      dt_imm = tail_imm->header.stamp.toSec() - last_lidar_end_time_imm;
    }
    else
    {
      dt_imm = tail_imm->header.stamp.toSec() - head_imm->header.stamp.toSec();
    }
    
    in_imm.acc = acc_avr_imm;
    in_imm.gyro = angvel_avr_imm;
    Q_imm.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q_imm.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q_imm.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q_imm.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state_imm.predict_imm(dt_imm, Q_imm, in_imm, state_points_imm, cov_imm);
    state_points_imm = kf_state_imm.get_x_imm();
    cov_imm = kf_state_imm.get_P_imm();

    /* save the poses at each IMU measurements */
    imu_state_imm = kf_state_imm.get_x_imm();

    angvel_last_imm = angvel_avr_imm - imu_state_imm.bg;
    acc_s_last_imm  = imu_state_imm.rot * (acc_avr_imm - imu_state_imm.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last_imm[i] += imu_state_imm.grav[i];
    }
    double &&offs_t_imm = tail_imm->header.stamp.toSec() - pcl_beg_time_imm;
    IMUpose_imm.push_back(set_pose6d(offs_t_imm, acc_s_last_imm, angvel_last_imm, imu_state_imm.vel, imu_state_imm.pos, imu_state_imm.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note_imm = pcl_end_time_imm > imu_end_time_imm ? 1.0 : -1.0;
  dt_imm = note_imm * (pcl_end_time_imm - imu_end_time_imm);

  kf_state_imm.predict_imm(dt_imm, Q_imm, in_imm, state_points_imm, cov_imm);
  state_points_imm = kf_state_imm.get_x_imm();
  cov_imm = kf_state_imm.get_P_imm();
  // cout << cov_imm << "IMM_cov" << endl;

  imu_state_imm = kf_state_imm.get_x_imm();
  last_imu_imm = meas_imm.imu.back();
  last_lidar_end_time_imm = pcl_end_time_imm;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out_imm.points.begin() == pcl_out_imm.points.end()) return;
  // ROS_WARN("debuging");
  auto it_pcl_imm = pcl_out_imm.points.end() - 1;
  for (auto it_kp_imm = IMUpose_imm.end() - 1; it_kp_imm != IMUpose_imm.begin(); it_kp_imm--)
  {
    // ROS_WARN("Really?");
    auto head_imm = it_kp_imm - 1;
    auto tail_imm = it_kp_imm;
    R_imu_imm<<MAT_FROM_ARRAY(head_imm->rot);
    vel_imu_imm<<VEC_FROM_ARRAY(head_imm->vel);
    pos_imu_imm<<VEC_FROM_ARRAY(head_imm->pos);
    acc_imu_imm<<VEC_FROM_ARRAY(tail_imm->acc);
    angvel_avr_imm<<VEC_FROM_ARRAY(tail_imm->gyr);

    for(; it_pcl_imm->curvature / double(1000) > head_imm->offset_time; it_pcl_imm --)
    {
      dt_imm = it_pcl_imm->curvature / double(1000) - head_imm->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i_imm(R_imu_imm * Exp(angvel_avr_imm, dt_imm));
      
      V3D P_i_imm(it_pcl_imm->x, it_pcl_imm->y, it_pcl_imm->z);
      V3D T_ei_imm(pos_imu_imm + vel_imu_imm * dt_imm + 0.5 * acc_imu_imm * dt_imm * dt_imm - imu_state_imm.pos);
      V3D P_compensate_imm = imu_state_imm.offset_R_L_I.conjugate() * (imu_state_imm.rot.conjugate() * (R_i_imm * (imu_state_imm.offset_R_L_I * P_i_imm + imu_state_imm.offset_T_L_I) + T_ei_imm) - imu_state_imm.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl_imm->x = P_compensate_imm(0);
      it_pcl_imm->y = P_compensate_imm(1);
      it_pcl_imm->z = P_compensate_imm(2);

      if (it_pcl_imm == pcl_out_imm.points.begin()) break;
    }
  }
}

void ImuProcess::IMU_init_imm(const MeasureGroup &meas_imm, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, int &N_imm)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc_imm, cur_gyr_imm;
  
  if (b_first_frame_imm)
  {
    Reset_imm();
    N_imm = 1;
    b_first_frame_imm = false;
    const auto &imu_acc_imm = meas_imm.imu.front()->linear_acceleration;
    const auto &gyr_acc_imm = meas_imm.imu.front()->angular_velocity;
    mean_acc_imm << imu_acc_imm.x, imu_acc_imm.y, imu_acc_imm.z;
    mean_gyr_imm << gyr_acc_imm.x, gyr_acc_imm.y, gyr_acc_imm.z;
    first_lidar_time_imm = meas_imm.lidar_beg_time;
  }

  for (const auto &imu_imm : meas_imm.imu)
  {
    const auto &imu_acc_imm = imu_imm->linear_acceleration;
    const auto &gyr_acc_imm = imu_imm->angular_velocity;
    cur_acc_imm << imu_acc_imm.x, imu_acc_imm.y, imu_acc_imm.z;
    cur_gyr_imm << gyr_acc_imm.x, gyr_acc_imm.y, gyr_acc_imm.z;

    mean_acc_imm      += (cur_acc_imm - mean_acc_imm) / N_imm;
    mean_gyr_imm     += (cur_gyr_imm - mean_gyr_imm) / N_imm;

    cov_acc_imm = cov_acc_imm * (N_imm - 1.0) / N_imm + (cur_acc_imm - mean_acc_imm).cwiseProduct(cur_acc_imm - mean_acc_imm) * (N_imm - 1.0) / (N_imm * N_imm);
    cov_gyr_imm = cov_gyr_imm * (N_imm - 1.0) / N_imm + (cur_gyr_imm - mean_acc_imm).cwiseProduct(cur_gyr_imm - mean_acc_imm) * (N_imm - 1.0) / (N_imm * N_imm);

    N_imm ++;
  }
  state_ikfom init_state_imm = kf_state_imm.get_x_imm();
  init_state_imm.grav = S2(- mean_acc_imm / mean_acc_imm.norm() * G_m_s2);
  init_state_imm.bg  = mean_gyr_imm;
  init_state_imm.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state_imm.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state_imm.change_x_imm(init_state_imm);


  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P_imm = kf_state_imm.get_P_imm();
  init_P_imm.setIdentity();
  init_P_imm(6,6) = init_P_imm(7,7) = init_P_imm(8,8) = 0.00001;
  init_P_imm(9,9) = init_P_imm(10,10) = init_P_imm(11,11) = 0.00001;
  init_P_imm(15,15) = init_P_imm(16,16) = init_P_imm(17,17) = 0.0001;
  init_P_imm(18,18) = init_P_imm(19,19) = init_P_imm(20,20) = 0.001;
  init_P_imm(21,21) = init_P_imm(22,22) = 0.00001; 
  kf_state_imm.change_P_imm(init_P_imm);
  last_imu_imm = meas_imm.imu.back();

}

// void ImuProcess::Process_imm(const MeasureGroup &meas_imm,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, PointCloudXYZI::Ptr cur_pcl_un_imm)
void ImuProcess::Process_imm(const MeasureGroup &meas_imm,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_imm, PointCloudXYZI::Ptr cur_pcl_un_imm, state_ikfom &state_points_imm, Matrix<double, 23, 23>  &P_imm_prev)
{
  if(meas_imm.imu.empty()) {return;};
  ROS_ASSERT(meas_imm.lidar != nullptr);

  if (imu_need_init_imm)
  {
    /// The very first lidar frame
    IMU_init_imm(meas_imm, kf_state_imm, init_iter_num_imm);

    imu_need_init_imm = true;
    
    last_imu_imm   = meas_imm.imu.back();

    state_ikfom imu_state_imm = kf_state_imm.get_x_imm();

    // ROS_WARN("what's wrong IMM");
    // cout << imu_state_imm << "IMM" << endl;
    if (init_iter_num_imm > MAX_INI_COUNT)
    {
      cov_acc_imm *= pow(G_m_s2 / mean_acc_imm.norm(), 2);
      imu_need_init_imm = false;
      P_init_imm = true;

      cov_acc_imm = cov_acc_scale;
      cov_gyr_imm = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }
  UndistortPcl_imm(meas_imm, kf_state_imm, *cur_pcl_un_imm, state_points_imm, P_imm_prev);
}


// void ImuProcess::UndistortPcl_ct(const MeasureGroup &meas_ct, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, PointCloudXYZI &pcl_out_ct)
void ImuProcess::UndistortPcl_ct(const MeasureGroup &meas_ct, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, PointCloudXYZI &pcl_out_ct, state_ikfom &state_points_ct, Matrix<double, 23, 23>  &P_ct_prev)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu_ct = meas_ct.imu;
  v_imu_ct.push_front(last_imu_ct);
  const double &imu_beg_time_ct = v_imu_ct.front()->header.stamp.toSec();
  const double &imu_end_time_ct = v_imu_ct.back()->header.stamp.toSec();
  const double &pcl_beg_time_ct = meas_ct.lidar_beg_time;
  const double &pcl_end_time_ct = meas_ct.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out_ct = *(meas_ct.lidar);
  sort(pcl_out_ct.points.begin(), pcl_out_ct.points.end(), time_list);

  /*** Initialize IMU pose ***/
  state_ikfom imu_state_ct = state_points_ct;
  if (P_init_ct)
  {
    cov_ct = kf_state_ct.get_P_ct();
    P_init_ct = false;
  }
  else
  {
    cov_ct = P_ct_prev;
  }


  IMUpose_ct.clear();
  IMUpose_ct.push_back(set_pose6d(0.0, acc_s_last_ct, angvel_last_ct, imu_state_ct.vel, imu_state_ct.pos, imu_state_ct.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr_ct, acc_avr_ct, acc_imu_ct, vel_imu_ct, pos_imu_ct;
  M3D R_imu_ct;

  double dt_ct = 0;

  input_ikfom in_ct;

  for (auto it_imu_ct = v_imu_ct.begin(); it_imu_ct < (v_imu_ct.end() - 1); it_imu_ct++)
  {
    auto &&head_ct = *(it_imu_ct);
    auto &&tail_ct = *(it_imu_ct + 1);
    
    if (tail_ct->header.stamp.toSec() < last_lidar_end_time_ct)    continue;
    
    angvel_avr_ct <<0.5 * (head_ct->angular_velocity.x + tail_ct->angular_velocity.x),
                     0.5 * (head_ct->angular_velocity.y + tail_ct->angular_velocity.y),
                     0.5 * (head_ct->angular_velocity.z + tail_ct->angular_velocity.z);
    acc_avr_ct   <<0.5 * (head_ct->linear_acceleration.x + tail_ct->linear_acceleration.x),
                    0.5 * (head_ct->linear_acceleration.y + tail_ct->linear_acceleration.y),
                    0.5 * (head_ct->linear_acceleration.z + tail_ct->linear_acceleration.z);

    acc_avr_ct     = acc_avr_ct * G_m_s2 / mean_acc_ct.norm(); // - state_inout.ba;

    if(head_ct->header.stamp.toSec() < last_lidar_end_time_ct)
    {
      dt_ct = tail_ct->header.stamp.toSec() - last_lidar_end_time_ct;
    }
    else
    {
      dt_ct = tail_ct->header.stamp.toSec() - head_ct->header.stamp.toSec();
    }
    
    in_ct.acc = acc_avr_ct;
    in_ct.gyro = angvel_avr_ct;
    Q_ct.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q_ct.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q_ct.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q_ct.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    kf_state_ct.predict_ct(dt_ct, Q_ct, in_ct, state_points_ct, cov_ct);
    cov_ct  = kf_state_ct.get_P_ct();

    in_acc = acc_avr_ct;
    in_angl = angvel_avr_ct;
    
    state_points_ct = kf_state_ct.get_x_ct();
   
    /* save the poses at each IMU measurements */
    imu_state_ct = kf_state_ct.get_x_ct();

    angvel_last_ct = angvel_avr_ct - imu_state_ct.bg;
    acc_s_last_ct  = imu_state_ct.rot * (acc_avr_ct - imu_state_ct.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last_ct[i] += imu_state_ct.grav[i];
    }
    double &&offs_t_ct = tail_ct->header.stamp.toSec() - pcl_beg_time_ct;
    IMUpose_ct.push_back(set_pose6d(offs_t_ct, acc_s_last_ct, angvel_last_ct, imu_state_ct.vel, imu_state_ct.pos, imu_state_ct.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note_ct = pcl_end_time_ct > imu_end_time_ct ? 1.0 : -1.0;
  dt_ct = note_ct * (pcl_end_time_ct - imu_end_time_ct);

  kf_state_ct.predict_ct(dt_ct, Q_ct, in_ct, state_points_ct, cov_ct);
  state_points_ct = kf_state_ct.get_x_ct();
  cov_ct  = kf_state_ct.get_P_ct();

  imu_state_ct = kf_state_ct.get_x_ct();
  last_imu_ct = meas_ct.imu.back();
  last_lidar_end_time_ct = pcl_end_time_ct;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out_ct.points.begin() == pcl_out_ct.points.end()) return;
  // ROS_WARN("debuging");
  auto it_pcl_ct = pcl_out_ct.points.end() - 1;
  for (auto it_kp_ct = IMUpose_ct.end() - 1; it_kp_ct != IMUpose_ct.begin(); it_kp_ct--)
  {
    // ROS_WARN("Really?");
    auto head_ct = it_kp_ct - 1;
    auto tail_ct = it_kp_ct;
    R_imu_ct<<MAT_FROM_ARRAY(head_ct->rot);
    vel_imu_ct<<VEC_FROM_ARRAY(head_ct->vel);
    pos_imu_ct<<VEC_FROM_ARRAY(head_ct->pos);
    acc_imu_ct<<VEC_FROM_ARRAY(tail_ct->acc);
    angvel_avr_ct<<VEC_FROM_ARRAY(tail_ct->gyr);

    for(; it_pcl_ct->curvature / double(1000) > head_ct->offset_time; it_pcl_ct --)
    {
      dt_ct = it_pcl_ct->curvature / double(1000) - head_ct->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i_ct(R_imu_ct * Exp(angvel_avr_ct, dt_ct));
      
      V3D P_i_ct(it_pcl_ct->x, it_pcl_ct->y, it_pcl_ct->z);
      V3D T_ei_ct(pos_imu_ct + vel_imu_ct * dt_ct + 0.5 * acc_imu_ct * dt_ct * dt_ct - imu_state_ct.pos);
      V3D P_compensate_ct = imu_state_ct.offset_R_L_I.conjugate() * (imu_state_ct.rot.conjugate() * (R_i_ct * (imu_state_ct.offset_R_L_I * P_i_ct + imu_state_ct.offset_T_L_I) + T_ei_ct) - imu_state_ct.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl_ct->x = P_compensate_ct(0);
      it_pcl_ct->y = P_compensate_ct(1);
      it_pcl_ct->z = P_compensate_ct(2);

      if (it_pcl_ct == pcl_out_ct.points.begin()) break;
    }
  }
}

void ImuProcess::IMU_init_ct(const MeasureGroup &meas_ct, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, int &N_ct)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc_ct, cur_gyr_ct;
  
  if (b_first_frame_ct)
  {
    Reset_ct();
    N_ct = 1;
    b_first_frame_ct = false;
    const auto &imu_acc_ct = meas_ct.imu.front()->linear_acceleration;
    const auto &gyr_acc_ct = meas_ct.imu.front()->angular_velocity;
    mean_acc_ct << imu_acc_ct.x, imu_acc_ct.y, imu_acc_ct.z;
    mean_gyr_ct << gyr_acc_ct.x, gyr_acc_ct.y, gyr_acc_ct.z;
    first_lidar_time_ct = meas_ct.lidar_beg_time;
  }

  for (const auto &imu_ct : meas_ct.imu)
  {
    const auto &imu_acc_ct = imu_ct->linear_acceleration;
    const auto &gyr_acc_ct = imu_ct->angular_velocity;
    cur_acc_ct << imu_acc_ct.x, imu_acc_ct.y, imu_acc_ct.z;
    cur_gyr_ct << gyr_acc_ct.x, gyr_acc_ct.y, gyr_acc_ct.z;

    mean_acc_ct      += (cur_acc_ct - mean_acc_ct) / N_ct;
    mean_gyr_ct     += (cur_gyr_ct - mean_gyr_ct) / N_ct;

    cov_acc_ct = cov_acc_ct * (N_ct - 1.0) / N_ct + (cur_acc_ct - mean_acc_ct).cwiseProduct(cur_acc_ct - mean_acc_ct) * (N_ct - 1.0) / (N_ct * N_ct);
    cov_gyr_ct = cov_gyr_ct * (N_ct - 1.0) / N_ct + (cur_gyr_ct - mean_acc_ct).cwiseProduct(cur_gyr_ct - mean_acc_ct) * (N_ct - 1.0) / (N_ct * N_ct);


    N_ct ++;
  }
  state_ikfom init_state_ct = kf_state_ct.get_x_ct();
  init_state_ct.grav = S2(- mean_acc_ct / mean_acc_ct.norm() * G_m_s2);

  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state_ct.bg  = mean_gyr_ct;
  init_state_ct.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state_ct.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state_ct.change_x_ct(init_state_ct);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P_ct = kf_state_ct.get_P_ct();
  init_P_ct.setIdentity();
  init_P_ct(6,6) = init_P_ct(7,7) = init_P_ct(8,8) = 0.00001;
  init_P_ct(9,9) = init_P_ct(10,10) = init_P_ct(11,11) = 0.00001;
  init_P_ct(15,15) = init_P_ct(16,16) = init_P_ct(17,17) = 0.0001;
  init_P_ct(18,18) = init_P_ct(19,19) = init_P_ct(20,20) = 0.001;
  init_P_ct(21,21) = init_P_ct(22,22) = 0.00001; 
  kf_state_ct.change_P_ct(init_P_ct);
  last_imu_ct = meas_ct.imu.back();
}

// void ImuProcess::Process_ct(const MeasureGroup &meas_ct,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, PointCloudXYZI::Ptr cur_pcl_un_ct, state_ikfom &state_points_ct)
void ImuProcess::Process_ct(const MeasureGroup &meas_ct,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ct, PointCloudXYZI::Ptr cur_pcl_un_ct, state_ikfom &state_points_ct, Matrix<double, 23, 23>  &P_ct_prev)
{
  if(meas_ct.imu.empty()) {return;};
  ROS_ASSERT(meas_ct.lidar != nullptr);

  if (imu_need_init_ct)
  {
    // The very first lidar frame
    IMU_init_ct(meas_ct, kf_state_ct, init_iter_num_ct);

    imu_need_init_ct = true;
    
    last_imu_ct   = meas_ct.imu.back();

    state_ikfom imu_state_ct = kf_state_ct.get_x_ct();

    if (init_iter_num_ct > MAX_INI_COUNT)
    {
      cov_acc_ct *= pow(G_m_s2 / mean_acc_ct.norm(), 2);
      imu_need_init_ct = false;
      P_init_ct = true;

      cov_acc_ct = cov_acc_scale;
      cov_gyr_ct = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }
  
  UndistortPcl_ct(meas_ct, kf_state_ct, *cur_pcl_un_ct, state_points_ct, P_ct_prev);
}

void ImuProcess::UndistortPcl_ca(const MeasureGroup &meas_ca, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ca, PointCloudXYZI &pcl_out_ca, state_ikfom &state_points_ca, Matrix<double, 23, 23>  &P_ca_prev)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu_ca = meas_ca.imu;
  v_imu_ca.push_front(last_imu_ca);
  const double &imu_beg_time_ca = v_imu_ca.front()->header.stamp.toSec();
  const double &imu_end_time_ca = v_imu_ca.back()->header.stamp.toSec();
  const double &pcl_beg_time_ca = meas_ca.lidar_beg_time;
  const double &pcl_end_time_ca = meas_ca.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out_ca = *(meas_ca.lidar);
  sort(pcl_out_ca.points.begin(), pcl_out_ca.points.end(), time_list);

  /*** Initialize IMU pose ***/
  state_ikfom imu_state_ca = state_points_ca;
  if (P_init_ca)
  {
    cov_ca = kf_state_ca.get_P_ca();
    P_init_ca = false;
  }
  else
  {
    cov_ca = P_ca_prev;
  }

  IMUpose_ca.clear();
  IMUpose_ca.push_back(set_pose6d(0.0, acc_s_last_ca, angvel_last_ca, imu_state_ca.vel, imu_state_ca.pos, imu_state_ca.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr_ca, acc_avr_ca, acc_imu_ca, vel_imu_ca, pos_imu_ca;
  M3D R_imu_ca;

  double dt_ca = 0;

  input_ikfom in_ca;

  for (auto it_imu_ca = v_imu_ca.begin(); it_imu_ca < (v_imu_ca.end() - 1); it_imu_ca++)
  {
    auto &&head_ca = *(it_imu_ca);
    auto &&tail_ca = *(it_imu_ca + 1);
    
    if (tail_ca->header.stamp.toSec() < last_lidar_end_time_ca)    continue;
    
    angvel_avr_ca <<0.5 * (head_ca->angular_velocity.x + tail_ca->angular_velocity.x),
                     0.5 * (head_ca->angular_velocity.y + tail_ca->angular_velocity.y),
                     0.5 * (head_ca->angular_velocity.z + tail_ca->angular_velocity.z);
    acc_avr_ca   <<0.5 * (head_ca->linear_acceleration.x + tail_ca->linear_acceleration.x),
                    0.5 * (head_ca->linear_acceleration.y + tail_ca->linear_acceleration.y),
                    0.5 * (head_ca->linear_acceleration.z + tail_ca->linear_acceleration.z);

    acc_avr_ca     = acc_avr_ca * G_m_s2 / mean_acc_ca.norm(); // - state_inout.ba;

    if(head_ca->header.stamp.toSec() < last_lidar_end_time_ca)
    {
      dt_ca = tail_ca->header.stamp.toSec() - last_lidar_end_time_ca;
    }
    else
    {
      dt_ca = tail_ca->header.stamp.toSec() - head_ca->header.stamp.toSec();
    }
    
    in_ca.acc = acc_avr_ca;
    in_ca.gyro = angvel_avr_ca;
    Q_ca.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q_ca.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q_ca.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q_ca.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state_ca.predict_ca(dt_ca, Q_ca, in_ca, state_points_ca, cov_ca);
    cov_ca = kf_state_ca.get_P_ca();

    in_acc = acc_avr_ca;
    in_angl = angvel_avr_ca;
    
    state_points_ca = kf_state_ca.get_x_ca();
   
    /* save the poses at each IMU measurements */
    imu_state_ca = kf_state_ca.get_x_ca();

    angvel_last_ca = angvel_avr_ca - imu_state_ca.bg;
    acc_s_last_ca  = imu_state_ca.rot * (acc_avr_ca - imu_state_ca.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last_ca[i] += imu_state_ca.grav[i];
    }
    double &&offs_t_ca = tail_ca->header.stamp.toSec() - pcl_beg_time_ca;
    IMUpose_ca.push_back(set_pose6d(offs_t_ca, acc_s_last_ca, angvel_last_ca, imu_state_ca.vel, imu_state_ca.pos, imu_state_ca.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note_ca = pcl_end_time_ca > imu_end_time_ca ? 1.0 : -1.0;
  dt_ca = note_ca * (pcl_end_time_ca - imu_end_time_ca);

  // kf_state_ca.predict_ca(dt_ca, Q_ca, in_ca, state_points_ca);
  kf_state_ca.predict_ca(dt_ca, Q_ca, in_ca, state_points_ca, cov_ca);
  cov_ca = kf_state_ca.get_P_ca();
  state_points_ca = kf_state_ca.get_x_ca();

  imu_state_ca = kf_state_ca.get_x_ca();
  last_imu_ca = meas_ca.imu.back();
  last_lidar_end_time_ca = pcl_end_time_ca;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out_ca.points.begin() == pcl_out_ca.points.end()) return;
  auto it_pcl_ca = pcl_out_ca.points.end() - 1;
  for (auto it_kp_ca = IMUpose_ca.end() - 1; it_kp_ca != IMUpose_ca.begin(); it_kp_ca--)
  {
    // ROS_WARN("Really?");
    auto head_ca = it_kp_ca - 1;
    auto tail_ca = it_kp_ca;
    R_imu_ca<<MAT_FROM_ARRAY(head_ca->rot);
    vel_imu_ca<<VEC_FROM_ARRAY(head_ca->vel);
    pos_imu_ca<<VEC_FROM_ARRAY(head_ca->pos);
    acc_imu_ca<<VEC_FROM_ARRAY(tail_ca->acc);
    angvel_avr_ca<<VEC_FROM_ARRAY(tail_ca->gyr);

    for(; it_pcl_ca->curvature / double(1000) > head_ca->offset_time; it_pcl_ca --)
    {
      dt_ca = it_pcl_ca->curvature / double(1000) - head_ca->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i_ca(R_imu_ca * Exp(angvel_avr_ca, dt_ca));
      
      V3D P_i_ca(it_pcl_ca->x, it_pcl_ca->y, it_pcl_ca->z);
      V3D T_ei_ca(pos_imu_ca + vel_imu_ca * dt_ca + 0.5 * acc_imu_ca * dt_ca * dt_ca - imu_state_ca.pos);
      V3D P_compensate_ca = imu_state_ca.offset_R_L_I.conjugate() * (imu_state_ca.rot.conjugate() * (R_i_ca * (imu_state_ca.offset_R_L_I * P_i_ca + imu_state_ca.offset_T_L_I) + T_ei_ca) - imu_state_ca.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl_ca->x = P_compensate_ca(0);
      it_pcl_ca->y = P_compensate_ca(1);
      it_pcl_ca->z = P_compensate_ca(2);

      if (it_pcl_ca == pcl_out_ca.points.begin()) break;
    }
  }
}

void ImuProcess::IMU_init_ca(const MeasureGroup &meas_ca, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ca, int &N_ca)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc_ca, cur_gyr_ca;
  
  if (b_first_frame_ca)
  {
    Reset_ca();
    N_ca = 1;
    b_first_frame_ca = false;
    const auto &imu_acc_ca = meas_ca.imu.front()->linear_acceleration;
    const auto &gyr_acc_ca = meas_ca.imu.front()->angular_velocity;
    mean_acc_ca << imu_acc_ca.x, imu_acc_ca.y, imu_acc_ca.z;
    mean_gyr_ca << gyr_acc_ca.x, gyr_acc_ca.y, gyr_acc_ca.z;
    first_lidar_time_ca = meas_ca.lidar_beg_time;
  }

  for (const auto &imu_ca : meas_ca.imu)
  {
    const auto &imu_acc_ca = imu_ca->linear_acceleration;
    const auto &gyr_acc_ca = imu_ca->angular_velocity;
    cur_acc_ca << imu_acc_ca.x, imu_acc_ca.y, imu_acc_ca.z;
    cur_gyr_ca << gyr_acc_ca.x, gyr_acc_ca.y, gyr_acc_ca.z;

    mean_acc_ca      += (cur_acc_ca - mean_acc_ca) / N_ca;
    mean_gyr_ca     += (cur_gyr_ca - mean_gyr_ca) / N_ca;

    cov_acc_ca = cov_acc_ca * (N_ca - 1.0) / N_ca + (cur_acc_ca - mean_acc_ca).cwiseProduct(cur_acc_ca - mean_acc_ca) * (N_ca - 1.0) / (N_ca * N_ca);
    cov_gyr_ca = cov_gyr_ca * (N_ca - 1.0) / N_ca + (cur_gyr_ca - mean_acc_ca).cwiseProduct(cur_gyr_ca - mean_acc_ca) * (N_ca - 1.0) / (N_ca * N_ca);

    N_ca ++;
  }
  state_ikfom init_state_ca = kf_state_ca.get_x_ca();
  init_state_ca.grav = S2(- mean_acc_ca / mean_acc_ca.norm() * G_m_s2);

  init_state_ca.bg  = mean_gyr_ca;
  init_state_ca.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state_ca.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state_ca.change_x_ca(init_state_ca);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P_ca = kf_state_ca.get_P_ca();
  init_P_ca.setIdentity();
  init_P_ca(6,6) = init_P_ca(7,7) = init_P_ca(8,8) = 0.00001;
  init_P_ca(9,9) = init_P_ca(10,10) = init_P_ca(11,11) = 0.00001;
  init_P_ca(15,15) = init_P_ca(16,16) = init_P_ca(17,17) = 0.0001;
  init_P_ca(18,18) = init_P_ca(19,19) = init_P_ca(20,20) = 0.001;
  init_P_ca(21,21) = init_P_ca(22,22) = 0.00001; 
  kf_state_ca.change_P_ca(init_P_ca);
  last_imu_ca = meas_ca.imu.back();
}

// void ImuProcess::Process_ca(const MeasureGroup &meas_ca,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ca, PointCloudXYZI::Ptr cur_pcl_un_ca, state_ikfom &state_points_ca)
void ImuProcess::Process_ca(const MeasureGroup &meas_ca,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state_ca, PointCloudXYZI::Ptr cur_pcl_un_ca, state_ikfom &state_points_ca, Matrix<double, 23, 23>  &P_ca_prev)
{
  if(meas_ca.imu.empty()) {return;};
  ROS_ASSERT(meas_ca.lidar != nullptr);

  if (imu_need_init_ca)
  {
    // The very first lidar frame
    IMU_init_ca(meas_ca, kf_state_ca, init_iter_num_ca);

    imu_need_init_ca = true;
    
    last_imu_ca   = meas_ca.imu.back();

    state_ikfom imu_state_ca = kf_state_ca.get_x_ca();
    if (init_iter_num_ca > MAX_INI_COUNT)
    {
      cov_acc_ca *= pow(G_m_s2 / mean_acc_ca.norm(), 2);
      imu_need_init_ca = false;
      P_init_ca = true;

      cov_acc_ca = cov_acc_scale;
      cov_gyr_ca = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl_ca(meas_ca, kf_state_ca, *cur_pcl_un_ca, state_points_ca, P_ca_prev);
}
