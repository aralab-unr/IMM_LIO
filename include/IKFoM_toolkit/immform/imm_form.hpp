#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP


#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"
#include "use-ikfom.hpp"

//#define USE_sparse


namespace esekfom {

using namespace Eigen;

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as a manifold.
template<typename S, typename M, int measurement_noise_dof = M::DOF>
struct share_datastruct
{
	bool valid;
	bool converge;
	M z;
	Eigen::Matrix<typename S::scalar, M::DOF, measurement_noise_dof> h_v;
	Eigen::Matrix<typename S::scalar, M::DOF, S::DOF> h_x;
	Eigen::Matrix<typename S::scalar, measurement_noise_dof, measurement_noise_dof> R;
};

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as an Eigen matrix whose dimension is changing
template<typename T>
struct dyn_share_datastruct
{
	bool valid;
	bool converge;
	Eigen::Matrix<T, Eigen::Dynamic, 1> z;
	Eigen::Matrix<T, Eigen::Dynamic, 1> h;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R;
};

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as a dynamic manifold whose dimension or type is changing
template<typename T>
struct dyn_runtime_share_datastruct
{
	bool valid;
	bool converge;
	//Z z;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R;
};

template<typename state, int process_noise_dof, typename input = state, typename measurement=state, int measurement_noise_dof=0>
class esekf{

	typedef esekf self;
	enum{
		n = state::DOF, m = state::DIM, l = measurement::DOF
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
	typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;
	typedef measurement measurementModel(state &, bool &);
	typedef measurement measurementModel_share(state &, share_datastruct<state, measurement, measurement_noise_dof> &);
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> measurementModel_dyn(state &, bool &);
	typedef void measurementModel_dyn_share(state &,  dyn_share_datastruct<scalar_type> &);
	typedef Eigen::Matrix<scalar_type ,l, n> measurementMatrix1(state &, bool&);
	typedef Eigen::Matrix<scalar_type , Eigen::Dynamic, n> measurementMatrix1_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,l, measurement_noise_dof> measurementMatrix2(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

	esekf(const state &x = state(),
		const cov  &P = cov::Identity()): x_(x), P_(P){
	#ifdef USE_sparse
		SparseMatrix<scalar_type> ref(n, n);
		ref.setIdentity();
		l_ = ref;
		f_x_2 = ref;
		f_x_1 = ref;
	#endif
	};

	//receive system-specific models and their differentions
	//for measurement as an Eigen matrix whose dimension is changing.
	//calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function (h_dyn_share_in).
	void init_dynamic(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel_dyn_share h_dyn_share_in, int maximum_iteration, scalar_type limit_vector[n])
	{
		f = f_in;
		f_x = f_x_in;
		f_w = f_w_in;
		h_dyn_share = h_dyn_share_in;

		fimm = f_in;
		f_ximm = f_x_in;
		f_wimm = f_w_in;
		h_dyn_share_imm = h_dyn_share_in;

		fca = f_in;
		f_xca = f_x_in;
		f_wca = f_w_in;
		h_dyn_share_ca = h_dyn_share_in;

		fct = f_in;
		f_xct = f_x_in;
		f_wct = f_w_in;
		h_dyn_share_ct = h_dyn_share_in;

		maximum_iter = maximum_iteration;
		for(int i=0; i<n; i++)
		{
			limit[i] = limit_vector[i];
		}

		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();

		x_imm.build_S2_state();
		x_imm.build_SO3_state();
		x_imm.build_vect_state();

		x_ca.build_S2_state();
		x_ca.build_SO3_state();
		x_ca.build_vect_state();

		x_ct.build_S2_state();
		x_ct.build_SO3_state();
		x_ct.build_vect_state();
	}


	//receive system-specific models and their differentions
	//for measurement as a dynamic manifold whose dimension  or type is changing.
	//calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function (h_dyn_share_in).
	//for any scenarios where it is needed

	// Prediction
	// void predict_ca(double &dt, processnoisecovariance &Q, const input &i_in, state &state_points_ca){
	void predict_ca(double &dt, processnoisecovariance &Q, const input &i_in, state &state_points_ca, cov &P_ca_prev){		
		flatted_state f_ca = fca(x_ca, i_in);
		cov_ f_x_ca = f_xca(x_ca, i_in);
		cov f_x_final_ca;
		Matrix<scalar_type, m, process_noise_dof> f_w_ca = f_wca(x_ca, i_in);
		Matrix<scalar_type, n, process_noise_dof> f_w_final_ca;
		x_ca = state_points_ca;
		state x_before = x_ca;

		Eigen::MatrixXd F_a;      // State transition matrix
		Eigen::MatrixXd x_hat_minus_a;  // State vector [x, y, z, vx, vy, vz, roll, pitch, yaw]
		Eigen::Vector3d prevPose;
		Eigen::Vector3d prevVel;
		Eigen::Vector3d prevAccel;
		Eigen::Vector3d prevAngular;
		Eigen::MatrixXd x_prev;
		x_prev = Eigen::MatrixXd::Zero(12, 1);
		F_a = Eigen::Matrix<double, 12, 12>::Identity();
		F_a.block<3,3>(0,3) = Eigen::Matrix<double, 3,3>::Identity() * dt;
		F_a.block<3,3>(3,9) = Eigen::Matrix<double, 3,3>::Identity() * dt;
		F_a.block<3,3>(0,9) = Eigen::Matrix<double, 3,3>::Identity() * 0.5*dt*dt;

		prevPose  = x_ca.pos;
		prevVel   = x_ca.vel;
		prevAccel = i_in.acc;
		prevAngular = i_in.gyro;
		x_prev.block<3,1>(0,0) = prevPose;
		x_prev.block<3,1>(3,0) = prevVel;
		x_prev.block<3,1>(6,0) = prevAngular;
		x_prev.block<3,1>(9,0) = prevAccel;
		/* Prediction CA */
        x_hat_minus_a = F_a * x_prev;

		x_ca.oplus(f_ca, dt);

		x_ca.pos = x_hat_minus_a.block<3,1>(0,0);
		x_ca.vel = x_hat_minus_a.block<3,1>(3,0);

		F_x1_ca = cov::Identity();
		// for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) 
		for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_ca.vect_state.begin(); it != x_ca.vect_state.end(); it++) 
		{

			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;
			for(int i = 0; i < n; i++){
				for(int j=0; j<dof; j++)
				{f_x_final_ca(idx+j, i) = f_x_ca(dim+j, i);}	
			}
			for(int i = 0; i < process_noise_dof; i++){
				for(int j=0; j<dof; j++)
				{f_w_final_ca(idx+j, i) = f_w_ca(dim+j, i);}
			}
		}
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<std::pair<int, int> >::iterator it = x_ca.SO3_state.begin(); it != x_ca.SO3_state.end(); it++)
		{
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_ca(dim + i) * dt;
			}
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
		#ifdef USE_sparse
			res_temp_SO3 = res.toRotationMatrix();
			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					f_x_1_ca.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
				}
			}
		#else
			F_x1_ca.template block<3, 3>(idx, idx) = res.toRotationMatrix();
		#endif			
			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < n; i++){
				f_x_final_ca. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_ca. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < process_noise_dof; i++){
				f_w_final_ca. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_ca. template block<3, 1>(dim, i));
			}
		}
		
		
		Matrix<scalar_type, 2, 3> res_temp_S2;
		Matrix<scalar_type, 2, 2> res_temp_S2_;
		MTK::vect<3, scalar_type> seg_S2;
		// for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) 
		for (std::vector<std::pair<int, int> >::iterator it = x_ca.S2_state.begin(); it != x_ca.S2_state.end(); it++)
		{
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_S2(i) = f_ca(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			// x_.S2_Nx_yy(Nx, idx);
			x_ca.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, vec, idx);
		#ifdef USE_sparse
			res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;
			for(int i = 0; i < 2; i++){
				for(int j = 0; j < 2; j++){
					f_x_1_ca.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
				}
			}
		#else
			F_x1_ca.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;
		#endif

			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();
			
			for(int i = 0; i < n; i++){
				f_x_final_ca. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_ca. template block<3, 1>(dim, i));
				
			}
			for(int i = 0; i < process_noise_dof; i++){
				f_w_final_ca. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_ca. template block<3, 1>(dim, i));
			}
		}
	
	#ifdef USE_sparse
		f_x_1_ca.makeCompressed();
		spMt f_x2_ca = f_x_final.sparseView();
		spMt f_w1 = f_w_final_ca.sparseView();
		spMt xp = f_x_1_ca + f_x2_ca * dt;
		P_ca = xp * P_ca * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
	#else
		F_x1_ca += f_x_final_ca * dt;

		P_ca = (F_x1_ca) * P_ca * (F_x1_ca).transpose() + (dt * f_w_final_ca) * Q * (dt * f_w_final_ca).transpose();
	#endif
	}

	void predict_imm(double &dt, processnoisecovariance &Q, const input &i_in, state &state_points_imm, cov &P_imm_prev){

	// void predict_imm(double &dt, processnoisecovariance &Q, const input &i_in){
		flatted_state f_imm = fimm(x_imm, i_in);
		cov_ f_x_imm = f_ximm(x_imm, i_in);
		cov f_x_final_imm;
		Matrix<scalar_type, m, process_noise_dof> f_w_imm = f_wimm(x_imm, i_in);
		Matrix<scalar_type, n, process_noise_dof> f_w_final_imm;
		x_imm = state_points_imm;
		state x_before = x_imm;

		Eigen::MatrixXd F_v;      // State transition matrix
		Eigen::MatrixXd x_hat_minus_v;  // State vector [x, y, z, vx, vy, vz, roll, pitch, yaw]
		Eigen::Vector3d prevPose;
		Eigen::Vector3d prevVel;
		Eigen::Vector3d prevAccel;
		Eigen::Vector3d prevAngular;
		Eigen::MatrixXd x_prev;
		x_prev = Eigen::MatrixXd::Zero(6, 1);
		F_v = Eigen::Matrix<double, 6, 6>::Identity();
		F_v.block<3,3>(0,3) = Eigen::Matrix<double, 3,3>::Identity() * dt;

		prevPose = x_imm.pos;
		prevVel  = x_imm.vel;

		x_prev.block<3,1>(0,0) = prevPose;
		x_prev.block<3,1>(3,0) = prevVel;
        x_hat_minus_v = F_v * x_prev;

		x_imm.oplus(f_imm, dt);

		x_imm.pos = x_hat_minus_v.block<3,1>(0,0);
		x_imm.vel = x_hat_minus_v.block<3,1>(3,0);

		F_x1_imm = cov::Identity();
		// for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) 
		for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_imm.vect_state.begin(); it != x_imm.vect_state.end(); it++) 
		{

			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;
			for(int i = 0; i < n; i++){
				for(int j=0; j<dof; j++)
				{f_x_final_imm(idx+j, i) = f_x_imm(dim+j, i);}	
			}
			for(int i = 0; i < process_noise_dof; i++){
				for(int j=0; j<dof; j++)
				{f_w_final_imm(idx+j, i) = f_w_imm(dim+j, i);}
			}
		}
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		// for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) 
		for (std::vector<std::pair<int, int> >::iterator it = x_imm.SO3_state.begin(); it != x_imm.SO3_state.end(); it++)
		{
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_imm(dim + i) * dt;
			}
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
		#ifdef USE_sparse
			res_temp_SO3 = res.toRotationMatrix();
			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					f_x_1_imm.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
				}
			}
		#else
			F_x1_imm.template block<3, 3>(idx, idx) = res.toRotationMatrix();
		#endif			
			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < n; i++){
				f_x_final_imm. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_imm. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < process_noise_dof; i++){
				f_w_final_imm. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_imm. template block<3, 1>(dim, i));
			}
		}
		
		
		Matrix<scalar_type, 2, 3> res_temp_S2;
		Matrix<scalar_type, 2, 2> res_temp_S2_;
		MTK::vect<3, scalar_type> seg_S2;
		// for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) 
		for (std::vector<std::pair<int, int> >::iterator it = x_imm.S2_state.begin(); it != x_imm.S2_state.end(); it++)
		{
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_S2(i) = f_imm(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			// x_.S2_Nx_yy(Nx, idx);
			x_imm.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, vec, idx);
		#ifdef USE_sparse
			res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;
			for(int i = 0; i < 2; i++){
				for(int j = 0; j < 2; j++){
					f_x_1_imm.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
				}
			}
		#else
			F_x1_imm.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;
		#endif

			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();
			
			for(int i = 0; i < n; i++){
				f_x_final_imm. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_imm. template block<3, 1>(dim, i));
				
			}
			for(int i = 0; i < process_noise_dof; i++){
				f_w_final_imm. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_imm. template block<3, 1>(dim, i));
			}
		}
	
	#ifdef USE_sparse
		f_x_1_imm.makeCompressed();
		spMt f_x2_imm = f_x_final.sparseView();
		spMt f_w1 = f_w_final_imm.sparseView();
		spMt xp = f_x_1_imm + f_x2_imm * dt;
		P_imm = xp * P_imm * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
	#else
		F_x1_imm += f_x_final_imm * dt;
		P_imm = (F_x1_imm) * P_imm * (F_x1_imm).transpose() + (dt * f_w_final_imm) * Q * (dt * f_w_final_imm).transpose();
	#endif
	}

	// void predict_ct(double &dt, processnoisecovariance &Q, const input &i_in, state &state_points_ct){
	void predict_ct(double &dt, processnoisecovariance &Q, const input &i_in, state &state_points_ct, cov &P_ct_prev){
		flatted_state f_ct = fct(x_ct, i_in);
		cov_ f_x_ct = f_xct(x_ct, i_in);
		cov f_x_final_ct;
		Matrix<scalar_type, m, process_noise_dof> f_w_ct = f_wct(x_ct, i_in);
		Matrix<scalar_type, n, process_noise_dof> f_w_final_ct;
		x_ct = state_points_ct;
		state x_before = x_ct;

		Eigen::MatrixXd F_t;      // State transition matrix
		Eigen::MatrixXd x_hat_minus_t;  // State vector [x, y, z, vx, vy, vz, roll, pitch, yaw]
		Eigen::Vector3d prevPose;
		Eigen::Vector3d prevVel;
		Eigen::Vector3d prevAccel;
		Eigen::Vector3d prevAngular;
		Eigen::MatrixXd x_prev;
		x_prev = Eigen::MatrixXd::Zero(12, 1);
		F_t = Eigen::Matrix<double, 12, 12>::Identity();
		prevPose  = x_ct.pos;
		prevVel   = x_ct.vel;
		prevAccel = i_in.acc;
		prevAngular = i_in.gyro;

		x_prev.block<3,1>(0,0) = prevPose;
		x_prev.block<3,1>(3,0) = prevVel;
		x_prev.block<3,1>(6,0) = prevAngular;
		x_prev.block<3,1>(9,0) = prevAccel;

		// double Tx    = prevAngular(0)/M_PI * 180;
        // double Ty    = prevAngular(1)/M_PI * 180;
        // double Tz    = prevAngular(2)/M_PI * 180;
		double Tx    = prevAngular(0);
        double Ty    = prevAngular(1);
        double Tz    = prevAngular(2);
        double T_xyz = sqrt(Tx*Tx + Ty*Ty + Tz*Tz);

		double c1    = (cos(T_xyz * dt) - 1)/(T_xyz*T_xyz);
        double c2    = sin(T_xyz*dt)/T_xyz;
        double c3    = (1/(T_xyz*T_xyz))*(sin(T_xyz)*dt)/(T_xyz - dt);
        double d1    = Ty*Ty + Tz*Tz;
        double d2    = Tx*Tx + Tz*Tz;
        double d3    = Tx*Tx + Ty*Ty;
        F_t = Eigen::MatrixXd::Identity(12, 12);

		Eigen::MatrixXd A;
        A   = Eigen::MatrixXd::Zero(3, 3);
        A.block<1,3>(0,0) << c1*d1, -c2*Tz-c1*Tx*Ty, c2*Ty-c1*Tx*Tz;
        A.block<1,3>(1,0) << c2*Tz-c1*Tx*Ty, c1*d2, -c2*Tx-c1*Ty*Tz;
        A.block<1,3>(2,0) << -c2*Ty-c1*Tx*Tz, c2*Tx-c1*Ty*Tz, c1*d3;

		Eigen::MatrixXd B;
        B   = Eigen::MatrixXd::Zero(3, 3);
        B.block<1,3>(0,0) << c3*d1, c1*Tz-c3*Tx*Ty, -c1*Ty-c3*Tx*Tz;
        B.block<1,3>(1,0) << -c1*Tz-c3*Tx*Ty, c3*d2, c1*Tx-c3*Ty*Tz;
        B.block<1,3>(2,0) << c1*Ty-c3*Tx*Tz, -c1*Tx-c3*Ty*Tz, c3*d3;

		F_t.block<3,3>(0,3) = B;
        F_t.block<3,3>(3,3) += A;
		
		// Predict Kalman CT
		x_hat_minus_t = F_t * x_prev; 

		x_ct.oplus(f_ct, dt);

		x_ct.pos = x_hat_minus_t.block<3,1>(0,0);
		x_ct.vel = x_hat_minus_t.block<3,1>(3,0);


		F_x1_ct = cov::Identity();
		for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_ct.vect_state.begin(); it != x_ct.vect_state.end(); it++) 
		{

			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;
			for(int i = 0; i < n; i++){
				for(int j=0; j<dof; j++)
				{f_x_final_ct(idx+j, i) = f_x_ct(dim+j, i);}	
			}
			for(int i = 0; i < process_noise_dof; i++){
				for(int j=0; j<dof; j++)
				{f_w_final_ct(idx+j, i) = f_w_ct(dim+j, i);}
			}
		}
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<std::pair<int, int> >::iterator it = x_ct.SO3_state.begin(); it != x_ct.SO3_state.end(); it++)
		{
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_ct(dim + i) * dt;
			}
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
		#ifdef USE_sparse
			res_temp_SO3 = res.toRotationMatrix();
			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					f_x_1_ct.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
				}
			}
		#else
			F_x1_ct.template block<3, 3>(idx, idx) = res.toRotationMatrix();
		#endif			
			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < n; i++){
				f_x_final_ct. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_ct. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < process_noise_dof; i++){
				f_w_final_ct. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_ct. template block<3, 1>(dim, i));
			}
		}
		
		
		Matrix<scalar_type, 2, 3> res_temp_S2;
		Matrix<scalar_type, 2, 2> res_temp_S2_;
		MTK::vect<3, scalar_type> seg_S2;
		for (std::vector<std::pair<int, int> >::iterator it = x_ct.S2_state.begin(); it != x_ct.S2_state.end(); it++)
		{
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_S2(i) = f_ct(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_ct.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, vec, idx);
		#ifdef USE_sparse
			res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;
			for(int i = 0; i < 2; i++){
				for(int j = 0; j < 2; j++){
					f_x_1_ct.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
				}
			}
		#else
			F_x1_ct.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;
		#endif

			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();
			
			for(int i = 0; i < n; i++){
				f_x_final_ct. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_ct. template block<3, 1>(dim, i));
				
			}
			for(int i = 0; i < process_noise_dof; i++){
				f_w_final_ct. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_ct. template block<3, 1>(dim, i));
			}
		}
	
	#ifdef USE_sparse
		f_x_1_ct.makeCompressed();
		spMt f_x2_ct = f_x_final_ct.sparseView();
		spMt f_w1 = f_w_final_ct.sparseView();
		spMt xp = f_x_1_ct + f_x2_ct * dt;
		P_ct = xp * P_ct * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
	#else

		F_x1_ct += f_x_final_ct * dt;
		P_ct = (F_x1_ct) * P_ct * (F_x1_ct).transpose() + (dt * f_w_final_ct) * Q * (dt * f_w_final_ct).transpose();
	#endif
	}

	//iterated error state EKF update modified for one specific system.
	void update_imm(double R, double &solve_time) {
		
		dyn_share_datastruct<scalar_type> dyn_share_imm;
		dyn_share_imm.valid = true;
		dyn_share_imm.converge = true;
		int t = 0;
		state x_propagated = x_imm;
		cov P_propagated = P_imm;
		int dof_Measurement; 

		Matrix<scalar_type, n, 1> K_h;
		Matrix<scalar_type, n, n> K_x; 

		vectorized_state dx_new = vectorized_state::Zero();
		for(int i=-1; i<maximum_iter; i++)
		{
			dyn_share_imm.valid = true;
			h_dyn_share_imm(x_imm, dyn_share_imm);

			if(! dyn_share_imm.valid)
			{
				continue; 
			}

			//Matrix<scalar_type, Eigen::Dynamic, 1> h = h_dyn_share(x_, dyn_share);
			#ifdef USE_sparse
				spMt h_x_imm = dyn_share_imm.h_x.sparseView();
			#else
				Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_imm = dyn_share_imm.h_x;
			#endif
			double solve_start = omp_get_wtime();
			dof_Measurement = h_x_imm.rows();
			
			vectorized_state dx;
			x_imm.boxminus(dx, x_propagated);
			
			dx_new = dx;

			P_imm = P_propagated;
			
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_imm.SO3_state.begin(); it != x_imm.SO3_state.end(); it++) 
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);;
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int i = 0; i < n; i++){
					P_imm. template block<3, 1>(idx, i) = res_temp_SO3 * (P_imm. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < n; i++){
					P_imm. template block<1, 3>(i, idx) =(P_imm. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}

			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_imm.S2_state.begin(); it != x_imm.S2_state.end(); it++) 
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_imm.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int i = 0; i < n; i++){
					P_imm. template block<2, 1>(idx, i) = res_temp_S2 * (P_imm. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < n; i++){
					P_imm. template block<1, 2>(i, idx) = (P_imm. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			if(n > dof_Measurement)
			{
				Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur_imm = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
				h_x_cur_imm.topLeftCorner(dof_Measurement, 12) = h_x_imm;


				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_imm * h_x_cur_imm.transpose() * (h_x_cur_imm * P_imm * h_x_cur_imm.transpose()/R + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse()/R;
				K_h = K_ * dyn_share_imm.h;
				K_x = K_ * h_x_cur_imm;
			}
			else
			{
			#ifdef USE_sparse 
				spMt A = h_x_imm.transpose() * h_x_imm;
				cov P_temp = (P_imm/R).inverse();
				P_temp. template block<12, 12>(0, 0) += A;
				P_temp = P_temp.inverse();
				K_ = P_temp. template block<n, 12>(0, 0) * h_x_imm.transpose();
				K_x = cov::Zero();
				K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;
			#else
				cov P_temp = (P_imm/R).inverse(); // R = 0.001;
				Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_imm.transpose() * h_x_imm;  // Calculate HTH
				P_temp. template block<12, 12>(0, 0) += HTH;
				cov P_inv = P_temp.inverse();
				K_h = P_inv. template block<n, 12>(0, 0) * h_x_imm.transpose() * dyn_share_imm.h;
				K_x.setZero(); // = cov::Zero();
				K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;
			#endif 
			}
			Eigen::MatrixXd H_a;      // Measurement matrix
			Eigen::MatrixXd P_a;
			H_a = Eigen::MatrixXd::Zero(3, 3);
			H_a.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
			P_a = P_imm.template block<3,3>(0, 0);
			
			Eigen::MatrixXd S_a = (H_a * P_a * H_a.transpose());
		
			// Lfun_imm = (1/sqrt(abs(2*M_PI * S_a.determinant())) * exp((-0.5 * dx_new.transpose()*dx_new)(0)));

			//K_x = K_ * h_x_;
			Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; 
			Lfun_imm = (1/sqrt(abs(2*M_PI * S_a.determinant())) * exp((-0.5 * dx_.transpose()*dx_)(0)));
			state x_before = x_imm;
			x_imm.boxplus(dx_);

			dyn_share_imm.converge = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) > limit[i])
				{
					dyn_share_imm.converge = false;
					break;
				}
			}
			if(dyn_share_imm.converge) t++;
			
			if(!t && i == maximum_iter - 2)
			{
				dyn_share_imm.converge = true;
			}

			if(t > 1 || i == maximum_iter - 1)
			{
				L_imm = P_imm;
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				// for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) 
				for(typename std::vector<std::pair<int, int> >::iterator it = x_imm.SO3_state.begin(); it != x_imm.SO3_state.end(); it++) 
				{
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int i = 0; i < n; i++){
						L_imm. template block<3, 1>(idx, i) = res_temp_SO3 * (P_imm. template block<3, 1>(idx, i)); 
					}
						for(int i = 0; i < 12; i++){
							K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
						}
					for(int i = 0; i < n; i++){
						L_imm. template block<1, 3>(i, idx) = (L_imm. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_imm. template block<1, 3>(i, idx) = (P_imm. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				// for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) 
				for(typename std::vector<std::pair<int, int> >::iterator it = x_imm.S2_state.begin(); it != x_imm.S2_state.end(); it++) 
				{
					int idx = (*it).first;

					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}

					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_imm.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
					for(int i = 0; i < n; i++){
						L_imm. template block<2, 1>(idx, i) = res_temp_S2 * (P_imm. template block<2, 1>(idx, i)); 
					}
						for(int i = 0; i < 12; i++){
							K_x. template block<2, 1>(idx, i) = res_temp_S2 * (K_x. template block<2, 1>(idx, i));
						}
					for(int i = 0; i < n; i++){
						L_imm. template block<1, 2>(i, idx) = (L_imm. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_imm. template block<1, 2>(i, idx) = (P_imm. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
					P_imm = L_imm - K_x.template block<n, 12>(0, 0) * P_imm.template block<12, n>(0, 0);

				solve_time += omp_get_wtime() - solve_start;
				return;
			}
			solve_time += omp_get_wtime() - solve_start;
		}
	}

	//iterated error state EKF update modified for one specific system.
	void update_ca(double R, double &solve_time) {
		
		dyn_share_datastruct<scalar_type> dyn_share_ca;
		dyn_share_ca.valid = true;
		dyn_share_ca.converge = true;
		int t = 0;
		state x_propagated = x_ca;
		cov P_propagated = P_ca;
		int dof_Measurement; 

		Matrix<scalar_type, n, 1> K_h;
		Matrix<scalar_type, n, n> K_x; 

		vectorized_state dx_new = vectorized_state::Zero();
		for(int i=-1; i<maximum_iter; i++)
		{
			dyn_share_ca.valid = true;
			h_dyn_share_ca(x_ca, dyn_share_ca);

			if(! dyn_share_ca.valid)
			{
				continue; 
			}

			#ifdef USE_sparse
				spMt h_x_ca = dyn_share_ca.h_x.sparseView();
			#else
				Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_ca = dyn_share_ca.h_x;
			#endif
			double solve_start = omp_get_wtime();
			dof_Measurement = h_x_ca.rows();
			
			vectorized_state dx;
			x_ca.boxminus(dx, x_propagated);
			
			dx_new = dx;

			P_ca = P_propagated;
			
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			// for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) 
			for (std::vector<std::pair<int, int> >::iterator it = x_ca.SO3_state.begin(); it != x_ca.SO3_state.end(); it++) 
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);;
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int i = 0; i < n; i++){
					P_ca. template block<3, 1>(idx, i) = res_temp_SO3 * (P_ca. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < n; i++){
					P_ca. template block<1, 3>(i, idx) =(P_ca. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}

			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			// for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) 
			for (std::vector<std::pair<int, int> >::iterator it = x_ca.S2_state.begin(); it != x_ca.S2_state.end(); it++) 
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_ca.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int i = 0; i < n; i++){
					P_ca. template block<2, 1>(idx, i) = res_temp_S2 * (P_ca. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < n; i++){
					P_ca. template block<1, 2>(i, idx) = (P_ca. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			if(n > dof_Measurement)
			{
				Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur_ca = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
				h_x_cur_ca.topLeftCorner(dof_Measurement, 12) = h_x_ca;


				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ca * h_x_cur_ca.transpose() * (h_x_cur_ca * P_ca * h_x_cur_ca.transpose()/R + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse()/R;
				K_h = K_ * dyn_share_ca.h;
				K_x = K_ * h_x_cur_ca;
			}
			else
			{
			#ifdef USE_sparse 
				spMt A = h_x_ca.transpose() * h_x_ca;
				cov P_temp = (P_ca/R).inverse();
				P_temp. template block<12, 12>(0, 0) += A;
				P_temp = P_temp.inverse();
				K_ = P_temp. template block<n, 12>(0, 0) * h_x_ca.transpose();
				K_x = cov::Zero();
				K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;
			#else
				cov P_temp = (P_ca/R).inverse(); // R = 0.001;
				Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_ca.transpose() * h_x_ca;  // Calculate HTH
				P_temp. template block<12, 12>(0, 0) += HTH;
				cov P_inv = P_temp.inverse();
				K_h = P_inv. template block<n, 12>(0, 0) * h_x_ca.transpose() * dyn_share_ca.h;
				K_x.setZero(); // = cov::Zero();
				K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;
			#endif 
			}
			// ROS_WARN("debugging");
			Eigen::MatrixXd H_v;      // Measurement matrix
			Eigen::MatrixXd P_v;
			H_v = Eigen::MatrixXd::Zero(3, 3);
			H_v.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
			P_v = P_ca.template block<3,3>(0, 0);
			Eigen::MatrixXd S_v = (H_v * P_v * H_v.transpose());

			// Lfun_v = (1/sqrt(abs(2*M_PI * S_v.determinant())) * exp((-0.5 * dx_new.transpose()*dx_new)(0)));
			// ROS_WARN("Lfun_v = %4f", Lfun_v);
			//K_x = K_ * h_x_;
			Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; 
			Lfun_v = (1/sqrt(abs(2*M_PI * S_v.determinant())) * exp((-0.5 * dx_.transpose()*dx_)(0)));

			state x_before = x_ca;
			x_ca.boxplus(dx_);

			dyn_share_ca.converge = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) > limit[i])
				{
					dyn_share_ca.converge = false;
					break;
				}
			}
			if(dyn_share_ca.converge) t++;
			
			if(!t && i == maximum_iter - 2)
			{
				dyn_share_ca.converge = true;
			}

			if(t > 1 || i == maximum_iter - 1)
			{
				L_ca = P_ca;
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				// for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) 
				for(typename std::vector<std::pair<int, int> >::iterator it = x_ca.SO3_state.begin(); it != x_ca.SO3_state.end(); it++) 
				{
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int i = 0; i < n; i++){
						L_ca. template block<3, 1>(idx, i) = res_temp_SO3 * (P_ca. template block<3, 1>(idx, i)); 
					}
						for(int i = 0; i < 12; i++){
							K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
						}
					for(int i = 0; i < n; i++){
						L_ca. template block<1, 3>(i, idx) = (L_ca. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_ca. template block<1, 3>(i, idx) = (P_ca. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				// for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) 
				for(typename std::vector<std::pair<int, int> >::iterator it = x_ca.S2_state.begin(); it != x_ca.S2_state.end(); it++) 
				{
					int idx = (*it).first;

					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}

					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_ca.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
					for(int i = 0; i < n; i++){
						L_ca. template block<2, 1>(idx, i) = res_temp_S2 * (P_ca. template block<2, 1>(idx, i)); 
					}
						for(int i = 0; i < 12; i++){
							K_x. template block<2, 1>(idx, i) = res_temp_S2 * (K_x. template block<2, 1>(idx, i));
						}
					for(int i = 0; i < n; i++){
						L_ca. template block<1, 2>(i, idx) = (L_ca. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_ca. template block<1, 2>(i, idx) = (P_ca. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
					P_ca = L_ca - K_x.template block<n, 12>(0, 0) * P_ca.template block<12, n>(0, 0);

				solve_time += omp_get_wtime() - solve_start;
				return;
			}
			solve_time += omp_get_wtime() - solve_start;
		}
	}

		//iterated error state EKF update modified for one specific system.
	void update_ct(double R, double &solve_time) {
		
		dyn_share_datastruct<scalar_type> dyn_share_ct;
		dyn_share_ct.valid = true;
		dyn_share_ct.converge = true;
		int t = 0;
		state x_propagated = x_ct;
		cov P_propagated = P_ct;
		int dof_Measurement; 

		Matrix<scalar_type, n, 1> K_h;
		Matrix<scalar_type, n, n> K_x; 

		vectorized_state dx_new = vectorized_state::Zero();
		for(int i=-1; i<maximum_iter; i++)
		{
			dyn_share_ct.valid = true;
			h_dyn_share_ct(x_ct, dyn_share_ct);

			if(! dyn_share_ct.valid)
			{
				continue; 
			}
			#ifdef USE_sparse
				spMt h_x_ct = dyn_share_ct.h_x.sparseView();
			#else
				Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_ct = dyn_share_ct.h_x;
			#endif
			double solve_start = omp_get_wtime();
			dof_Measurement = h_x_ct.rows();
			
			vectorized_state dx;
			x_ct.boxminus(dx, x_propagated);
			
			dx_new = dx;

			P_ct = P_propagated;
			
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;

			for (std::vector<std::pair<int, int> >::iterator it = x_ct.SO3_state.begin(); it != x_ct.SO3_state.end(); it++) 
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);;
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int i = 0; i < n; i++){
					P_ct. template block<3, 1>(idx, i) = res_temp_SO3 * (P_ct. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < n; i++){
					P_ct. template block<1, 3>(i, idx) =(P_ct. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}

			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_ct.S2_state.begin(); it != x_ct.S2_state.end(); it++) 
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_ct.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int i = 0; i < n; i++){
					P_ct. template block<2, 1>(idx, i) = res_temp_S2 * (P_ct. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < n; i++){
					P_ct. template block<1, 2>(i, idx) = (P_ct. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			if(n > dof_Measurement)
			{
				Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur_ct = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
				h_x_cur_ct.topLeftCorner(dof_Measurement, 12) = h_x_ct;


				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ct * h_x_cur_ct.transpose() * (h_x_cur_ct * P_ct * h_x_cur_ct.transpose()/R + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse()/R;
				K_h = K_ * dyn_share_ct.h;
				K_x = K_ * h_x_cur_ct;
			}
			else
			{
			#ifdef USE_sparse 
				spMt A = h_x_ct.transpose() * h_x_ct;
				cov P_temp = (P_ct/R).inverse();
				P_temp. template block<12, 12>(0, 0) += A;
				P_temp = P_temp.inverse();
				K_ = P_temp. template block<n, 12>(0, 0) * h_x_ct.transpose();
				K_x = cov::Zero();
				K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;
			#else
				cov P_temp = (P_ct/R).inverse(); // R = 0.001;
				Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_ct.transpose() * h_x_ct;  // Calculate HTH
				P_temp. template block<12, 12>(0, 0) += HTH;
				cov P_inv = P_temp.inverse();
				K_h = P_inv. template block<n, 12>(0, 0) * h_x_ct.transpose() * dyn_share_ct.h;
				K_x.setZero(); // = cov::Zero();
				K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;
			#endif 
			}
			
			// Modification
			Eigen::MatrixXd H_t;      // Measurement matrix
			Eigen::MatrixXd P_t;
			H_t = Eigen::MatrixXd::Zero(3, 3);
			H_t.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
			P_t = P_ct.template block<3,3>(0, 0);
			Eigen::MatrixXd S_t = (H_t * P_t * H_t.transpose());

			// Lfun_t = (1/sqrt(abs(2*M_PI * S_t.determinant())) * exp((-0.5 * dx_new.transpose()*dx_new)(0)));

			//K_x = K_ * h_x_;
			Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; 

			Lfun_t = (1/sqrt(abs(2*M_PI * S_t.determinant())) * exp((-0.5 * dx_.transpose()*dx_)(0)));


			state x_before = x_ct;
			x_ct.boxplus(dx_);

			dyn_share_ct.converge = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) > limit[i])
				{
					dyn_share_ct.converge = false;
					break;
				}
			}
			if(dyn_share_ct.converge) t++;
			
			if(!t && i == maximum_iter - 2)
			{
				dyn_share_ct.converge = true;
			}

			if(t > 1 || i == maximum_iter - 1)
			{
				L_ct = P_ct;
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				// for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) 
				for(typename std::vector<std::pair<int, int> >::iterator it = x_ct.SO3_state.begin(); it != x_ct.SO3_state.end(); it++) 
				{
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int i = 0; i < n; i++){
						L_ct. template block<3, 1>(idx, i) = res_temp_SO3 * (P_ct. template block<3, 1>(idx, i)); 
					}
						for(int i = 0; i < 12; i++){
							K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
						}
					for(int i = 0; i < n; i++){
						L_ct. template block<1, 3>(i, idx) = (L_ct. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_ct. template block<1, 3>(i, idx) = (P_ct. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				// for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) 
				for(typename std::vector<std::pair<int, int> >::iterator it = x_ct.S2_state.begin(); it != x_ct.S2_state.end(); it++) 
				{
					int idx = (*it).first;

					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}

					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_ct.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
					for(int i = 0; i < n; i++){
						L_ct. template block<2, 1>(idx, i) = res_temp_S2 * (P_ct. template block<2, 1>(idx, i)); 
					}
						for(int i = 0; i < 12; i++){
							K_x. template block<2, 1>(idx, i) = res_temp_S2 * (K_x. template block<2, 1>(idx, i));
						}
					for(int i = 0; i < n; i++){
						L_ct. template block<1, 2>(i, idx) = (L_ct. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_ct. template block<1, 2>(i, idx) = (P_ct. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
					P_ct = L_ct - K_x.template block<n, 12>(0, 0) * P_ct.template block<12, n>(0, 0);

				solve_time += omp_get_wtime() - solve_start;
				return;
			}
			solve_time += omp_get_wtime() - solve_start;
		}
		
	}


	void change_x(state &input_state)
	{
		x_ = input_state;
		if((!x_.vect_state.size())&&(!x_.SO3_state.size())&&(!x_.S2_state.size()))
		{
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
		}
	}

	void change_x_imm(state &input_state)
	{
		x_imm = input_state;
		if((!x_imm.vect_state.size())&&(!x_imm.SO3_state.size())&&(!x_imm.S2_state.size()))
		{
			x_imm.build_S2_state();
			x_imm.build_SO3_state();
			x_imm.build_vect_state();
		}
	}

	void change_x_ca(state &input_state)
	{
		x_ca = input_state;
		if((!x_ca.vect_state.size())&&(!x_ca.SO3_state.size())&&(!x_ca.S2_state.size()))
		{
			x_ca.build_S2_state();
			x_ca.build_SO3_state();
			x_ca.build_vect_state();
		}
	}

	void change_x_ct(state &input_state)
	{
		x_ct = input_state;
		if((!x_ct.vect_state.size())&&(!x_ct.SO3_state.size())&&(!x_ct.S2_state.size()))
		{
			x_ct.build_S2_state();
			x_ct.build_SO3_state();
			x_ct.build_vect_state();
		}
	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	void change_P_imm(cov &input_cov)
	{
		P_imm = input_cov;
	}

	void change_P_ca(cov &input_cov)
	{
		P_ca = input_cov;
	}

	void change_P_ct(cov &input_cov)
	{
		P_ct = input_cov;
	}

	const double& get_Lfun_v() const {
		return Lfun_v;
	}

	const double& get_Lfun_imm() const {
		return Lfun_imm;
	}

	const double& get_Lfun_t() const {
		return Lfun_t;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}

	const state& get_x_imm() const {
		return x_imm;
	}

	const state& get_x_ca() const {
		return x_ca;
	}

	const state& get_x_ct() const {
		return x_ct;
	}

	const cov& get_P_imm() const {
		return P_imm;
	}

	const cov& get_P_ca() const {
		return P_ca;
	}


	const cov& get_P_ct() const {
		return P_ct;
	}


private:
	state x_;
	state x_imm;
	state x_ca;
	state x_ct;
	measurement m_;

	cov P_prev;
	cov P_imm_prev;
	cov P_ct_prev;

	cov P_;
	cov P_imm;
	cov P_ca;
	cov P_ct;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	spMt f_x_1_imm;
	spMt f_x_2_imm;
	spMt f_x_1_ca;
	spMt f_x_2_ca;
	spMt f_x_1_ct;
	spMt f_x_2_ct;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov F_x1_imm = cov::Identity();
	cov F_x2_imm = cov::Identity();
	cov F_x1_ca = cov::Identity();
	cov F_x2_ca = cov::Identity();
	cov F_x1_ct = cov::Identity();
	cov F_x2_ct = cov::Identity();
	cov L_ = cov::Identity();
	cov L_imm = cov::Identity();
	cov L_ca = cov::Identity();
	cov L_ct = cov::Identity();

	processModel *f;
	processMatrix1 *f_x;
	processMatrix2 *f_w;

	processModel *fimm;
	processMatrix1 *f_ximm;
	processMatrix2 *f_wimm;

	processModel *fca;
	processMatrix1 *f_xca;
	processMatrix2 *f_wca;

	measurementModel *h;
	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;

	measurementModel *h_imm;
	measurementMatrix1 *h_x_imm;
	measurementMatrix2 *h_v_imm;

	measurementModel *h_ca;
	measurementMatrix1 *h_x_ca;
	measurementMatrix2 *h_v_ca;

	measurementModel_dyn *h_dyn;
	measurementMatrix1_dyn *h_x_dyn;
	measurementMatrix2_dyn *h_v_dyn;

	measurementModel_dyn *h_dyn_imm;
	measurementMatrix1_dyn *h_x_dyn_imm;
	measurementMatrix2_dyn *h_v_dyn_imm;

	measurementModel_dyn *h_dyn_ca;
	measurementMatrix1_dyn *h_x_dyn_ca;
	measurementMatrix2_dyn *h_v_dyn_ca;

	measurementModel_share *h_share;
	measurementModel_dyn_share *h_dyn_share;

	measurementModel_share *h_share_imm;
	measurementModel_dyn_share *h_dyn_share_imm;

	measurementModel_share *h_share_ca;
	measurementModel_dyn_share *h_dyn_share_ca;

	processModel *fct;
	processMatrix1 *f_xct;
	processMatrix2 *f_wct;

	measurementModel *h_ct;
	measurementMatrix1 *h_x_ct;
	measurementMatrix2 *h_v_ct;

	measurementModel_dyn *h_dyn_ct;
	measurementMatrix1_dyn *h_x_dyn_ct;
	measurementMatrix2_dyn *h_v_dyn_ct;

	measurementModel_share *h_share_ct;
	measurementModel_dyn_share *h_dyn_share_ct;
	double Lfun_v;
	double Lfun_t;
	double Lfun_a;
	double Lfun_imm;

	int maximum_iter = 0;
	scalar_type limit[n];
	
	template <typename T>
    T check_safe_update( T _temp_vec )
    {
        T temp_vec = _temp_vec;
        if ( std::isnan( temp_vec(0, 0) ) )
        {
            temp_vec.setZero();
            return temp_vec;
        }
        double angular_dis = temp_vec.block( 0, 0, 3, 1 ).norm() * 57.3;
        double pos_dis = temp_vec.block( 3, 0, 3, 1 ).norm();
        if ( angular_dis >= 20 || pos_dis > 1 )
        {
            printf( "Angular dis = %.2f, pos dis = %.2f\r\n", angular_dis, pos_dis );
            temp_vec.setZero();
        }
        return temp_vec;
    }
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
