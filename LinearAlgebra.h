#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace LinearAlgebra
{
	enum Axis { X, Y, Z, Quaternions };
	enum LinearSolver { SVD, HouseHolderQR, ColPivHouseHolderQR, FullPivHouseHolderQR, CholeskyDec, Inversion };

	typedef unsigned int uint;
	typedef const int cint;
	typedef const unsigned int cuint;
	typedef const double cdouble;

	#define M_PI 3.1415926535897932384626433832795028841

	template<typename T> inline const T sum(const std::vector<T>& v)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty.");

		T sum = v[0];
		for (size_t i = 1; i < v.size(); ++i)
			sum += v[i];
		return sum;
	}
	
	template<typename T> inline const T sumSquares(const std::vector<T>& v)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty.");

		T sum = (v[0] * v[0]);
		for (size_t i = 1; i < v.size(); ++i)
			sum += (v[i] * v[i]);
		return sum;
	}
	
	template<typename T> inline const std::vector<T> addition(const std::vector<T>& v, const std::vector<T>& w)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty");

		const size_t N = v.size();
		if (w.size() != N)
			throw std::invalid_argument("input vectors dimension mismatch");

		std::vector<T> r(N);
		for (size_t i = 0; i < N; ++i)
			r[i] = v[i] + w[i];
		return r;
	}
	
	template<typename T> inline const std::vector<T> addition(const std::vector<T>& v, const T& w)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty");

		const size_t N = v.size();
		std::vector<T> r(N);
		for (size_t i = 0; i < N; ++i)
			r[i] = v[i] + w;
		return r;
	}
	
	template<typename T> inline const std::vector<T> subtraction(const std::vector<T>& v, const std::vector<T>& w)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty");

		const size_t N = v.size();
		if (w.size() != N)
			throw std::invalid_argument("input vectors dimension mismatch");

		std::vector<T> r(N);
		for (size_t i = 0; i < N; ++i)
			r[i] = v[i] - w[i];
		return r;
	}
	
	template<typename T> inline const std::vector<T> multiplication(const std::vector<T>& v, const std::vector<T>& w)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty");

		const size_t N = v.size();
		if (w.size() != N)
			throw std::invalid_argument("input vectors dimension mismatch");

		std::vector<T> r(N);
		for (size_t i = 0; i < N; ++i)
			r[i] = v[i] * w[i];
		return r;
	}
	
	template<typename T> inline const std::vector<T> multiplication(const std::vector<T>& v, const T& w)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty");

		const size_t N = v.size();
		std::vector<T> r(N);
		for (size_t i = 0; i < N; ++i)
			r[i] = v[i] * w;
		return r;
	}
	
	template<typename T> inline const std::vector<T> division(const std::vector<T>& v, const std::vector<T>& w)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty");

		const size_t N = v.size();
		if (w.size() != N)
			throw std::invalid_argument("input vectors dimension mismatch");

		std::vector<T> r(N);
		for (size_t i = 0; i < N; ++i)
		{
			if (w[i] != 0.0)
				r[i] = v[i] / w[i];
			else
				r[i] = std::copysign(std::numeric_limits<T>::max(), v[i]);
		}
		return r;
	}
	
	template<typename T> inline const std::vector<T> division(const std::vector<T>& v, const T& w)
	{
		if (v.size() == 0)
			throw std::invalid_argument("vector is empty");

		const size_t N = v.size();
		std::vector<T> r(N);
		for (size_t i = 0; i < N; ++i)
		{
			if (w != 0.0)
				r[i] = v[i] / w;
			else
				r[i] = std::copysign(std::numeric_limits<T>::max(), v[i]);
		}
		return r;
	}
	
	template<typename T> inline const T mean(const std::vector<T>& v)
	{
		return sum<T>(v) / v.size();
	}
	
	template<typename T> inline const T variance(const std::vector<T>& v)
	{
		return sumSquares<T>(addition<T>(v, -mean<T>(v))) / v.size();
	}
	
	template<typename T> inline const T std(const std::vector<T>& v)
	{
		return sqrt(variance<T>(v));
	}

	cdouble randDouble(cdouble min, cdouble max);
	cdouble roundoff(cdouble x, cuint N);
	cdouble rSquared(const std::vector<double>& y, const std::vector<double>& f);
	const Eigen::Matrix3d normalizeMatrixToRot(const Eigen::Matrix3d& M);
	const Eigen::Vector4d normalizeQuatToRot(const Eigen::Vector4d& q);
	const Eigen::Matrix4d assembleRotoTranslation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	const Eigen::Vector4d quaternionInverse(const Eigen::Vector4d& q);
	const Eigen::Vector4d quaternionProduct(const Eigen::Vector4d& q, const Eigen::Vector4d& r);

	class RotationSequence
	{
	public:
		RotationSequence(Axis isQuaternion = Axis::Quaternions);
		RotationSequence(Axis first, Axis second, Axis third);
		RotationSequence(cuint first, cuint second, cuint third);
		RotationSequence(const RotationSequence& seq1);
		RotationSequence(const RotationSequence* seq1);

		RotationSequence& operator=(const RotationSequence& seq);
		friend bool operator!=(const RotationSequence& seq1, const RotationSequence& seq2);
		friend bool operator==(const RotationSequence& seq1, const RotationSequence& seq2) { return !(seq1 != seq2); }

		bool isQuaternion() const { return _isQuat; }
		Axis first() const { return _first; }
		Axis second() const { return _second; }
		Axis third() const { return _third; }
		const std::string firstToStr() const;
		const std::string secondToStr() const;
		const std::string thirdToStr() const;
		const cuint firstInt() const { return _firstInt; }
		const cuint secondInt() const { return _secondInt; }
		const cuint thirdInt() const { return _thirdInt; }
		Axis getNextAxis();
		const cuint getNextAxisInt();
		const std::vector<Axis> sequence() const { return std::vector<Axis>{ _first, _second, _third }; }
		const std::vector<uint> sequenceInt() const { return std::vector<uint>{ _firstInt, _secondInt, _thirdInt }; }
		void setFirst(Axis axis);
		void setFirst(cuint axis);
		void setSecond(Axis axis);
		void setSecond(cuint axis);
		void setThird(Axis axis);
		void setThird(cuint axis);
		void setNextAxis(Axis axis);
		void setNextAxis(cuint axis);
		const std::string toStr() const;

		static const RotationSequence strToSequence(const std::string& text, bool* ok = nullptr);

	private:
		bool _isQuat = false;
		Axis _first;
		Axis _second;
		Axis _third;
		uint _firstInt;
		uint _secondInt;
		uint _thirdInt;
		uint _cntSet = 0;
		uint _cntGet = 0;
	};
	
	namespace Rotations
	{
		static const RotationSequence XYZ = RotationSequence(Axis::X, Axis::Y, Axis::Z);
		static const RotationSequence XZY = RotationSequence(Axis::X, Axis::Z, Axis::Y);
		static const RotationSequence YXZ = RotationSequence(Axis::Y, Axis::X, Axis::Z);
		static const RotationSequence YZX = RotationSequence(Axis::Y, Axis::Z, Axis::X);
		static const RotationSequence ZXY = RotationSequence(Axis::Z, Axis::X, Axis::Y);
		static const RotationSequence ZYX = RotationSequence(Axis::Z, Axis::Y, Axis::X);
		static const RotationSequence XYX = RotationSequence(Axis::X, Axis::Y, Axis::X);
		static const RotationSequence XZX = RotationSequence(Axis::X, Axis::Z, Axis::X);
		static const RotationSequence YXY = RotationSequence(Axis::Y, Axis::X, Axis::Y);
		static const RotationSequence YZY = RotationSequence(Axis::Y, Axis::Z, Axis::Y);
		static const RotationSequence ZXZ = RotationSequence(Axis::Z, Axis::X, Axis::Z);
		static const RotationSequence ZYZ = RotationSequence(Axis::Z, Axis::Y, Axis::Z);
		static const RotationSequence Quat = RotationSequence(LinearAlgebra::Axis::Quaternions);
	}

	const Eigen::Vector3d eulToEul(const RotationSequence srcSeq, const RotationSequence dstSeq, const Eigen::Vector3d& eul);
	const Eigen::Matrix3d eulToRotmatrix(Axis first, Axis second, Axis third, const Eigen::Vector3d& eul);
	const Eigen::Matrix3d eulToRotmatrix(const RotationSequence seq, const Eigen::Vector3d& eul);
	const Eigen::Vector4d eulToQuaternion(Axis first, Axis second, Axis third, const Eigen::Vector3d& eul);
	const Eigen::Vector4d eulToQuaternion(const RotationSequence seq, const Eigen::Vector3d& eul);
	const Eigen::Matrix3d rotmatrix(Axis axis, cdouble a);
	const Eigen::Matrix3d rotmatrixX(cdouble a);
	const Eigen::Matrix3d rotmatrixY(cdouble a);
	const Eigen::Matrix3d rotmatrixZ(cdouble a);
	const Eigen::Vector3d rotmatrixToEul(Axis first, Axis second, Axis third, const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEul(const RotationSequence seq, const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulXYZ(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulXZY(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulYXZ(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulYZX(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulZXY(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulZYX(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulXYX(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulXZX(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulYXY(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulYZY(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulZXZ(const Eigen::Matrix3d& R);
	const Eigen::Vector3d rotmatrixToEulZYZ(const Eigen::Matrix3d& R);
	const Eigen::Vector4d rotmatrixToQuaternion(const Eigen::Matrix3d& R);
	const Eigen::Vector3d quaternionToEul(Axis first, Axis second, Axis third, const Eigen::Vector4d& q);
	const Eigen::Vector3d quaternionToEul(const RotationSequence seq, const Eigen::Vector4d& q);
	const Eigen::Matrix3d quaternionToRotmatrix(const Eigen::Vector4d& q);

	namespace CellAlignment
	{
		struct CellTransformation
		{
			CellTransformation() { }
			CellTransformation(const Eigen::Matrix4d& M, const Eigen::Vector3d& TCP) : M(M), TCP(TCP) { }
			Eigen::Matrix4d M;
			Eigen::Vector3d TCP;
		};
		const CellTransformation cellAlignmentAlgorithm(const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& r, cuint steadyStateN = 50, cdouble steadyStateThres = 0.01, cuint maxIter = 1000, const LinearSolver solver = SVD);
		const Eigen::Matrix4d estimateRotoTranslation(const Eigen::Vector3d& lmean, const Eigen::MatrixX3d& Y, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Matrix3d>& R, const Eigen::Vector3d& v);
		const Eigen::Vector3d estimateTCPTranslation(const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Matrix3d>& R, const Eigen::Matrix4d& M, const LinearSolver solver);
		const bool checkSteadyState(const Eigen::Matrix4d& Mold, const Eigen::Matrix4d& M, const Eigen::Vector3d& vold, const Eigen::Vector3d& v, cdouble threshold);
		const std::vector<Eigen::Vector3d> evaluateModel(const Eigen::Matrix4d& M, const Eigen::Vector3d& v, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& r);
		const double assessModel(const std::vector<Eigen::Vector3d>& l_fitted, const std::vector<Eigen::Vector3d>& l);
		const std::vector<std::pair<unsigned int, double>> getWorstFittingData(const std::vector<Eigen::Vector3d>& l_fitted, const std::vector<Eigen::Vector3d>& l_true, cdouble threshold, cuint N);
	}
}