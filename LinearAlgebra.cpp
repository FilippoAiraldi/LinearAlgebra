#include "LinearAlgebra.h"
#include <cmath>
#include <iomanip>

namespace LinearAlgebra
{
	cdouble randDouble(cdouble min, cdouble max)
	{
		if (min >= max)
			throw std::invalid_argument("invalid min and max arguments");
		double f = (double)rand() / RAND_MAX;
		return min + f * (max - min);
	}

	cdouble LinearAlgebra::roundoff(cdouble x, cuint N)
	{
		std::stringstream ss;
		ss << std::fixed << std::setprecision(N) << x;
		return std::stod(ss.str());
	}

	const Eigen::Matrix3d LinearAlgebra::normalizeMatrixToRot(const Eigen::Matrix3d& M)
	{
		Eigen::Vector3d X(M(0, 0), M(1, 0), M(2, 0));
		Eigen::Vector3d Y(M(0, 1), M(1, 1), M(2, 1));
		Eigen::Vector3d Z(M(0, 2), M(1, 2), M(2, 2));
		double exy = X(0) * Y(0) + X(1) * Y(1) + X(2) * Y(2);
		double exz = X(0) * Z(0) + X(1) * Z(1) + X(2) * Z(2);
		double eyz = Z(0) * Y(0) + Z(1) * Y(1) + Z(2) * Y(2);
		Eigen::Vector3d Xort(X(0) - exy / 2 * Y(0) - exz / 2 * Z(0),
			X(1) - exy / 2 * Y(1) - exz / 2 * Z(1),
			X(2) - exy / 2 * Y(2) - exz / 2 * Z(2));
		Eigen::Vector3d Yort(Y(0) - exy / 2 * X(0) - eyz / 2 * Z(0),
			Y(1) - exy / 2 * X(1) - eyz / 2 * Z(1),
			Y(2) - exy / 2 * X(2) - eyz / 2 * Z(2));
		Eigen::Vector3d Zort(Z(0) - exz / 2 * X(0) - eyz / 2 * Y(0),
			Z(1) - exz / 2 * X(1) - eyz / 2 * Y(1),
			Z(2) - exz / 2 * X(2) - eyz / 2 * Y(2));
		double Xnorm = sqrt(Xort(0) * Xort(0) + Xort(1) * Xort(1) + Xort(2) * Xort(2));
		double Ynorm = sqrt(Yort(0) * Yort(0) + Yort(1) * Yort(1) + Yort(2) * Yort(2));
		double Znorm = sqrt(Zort(0) * Zort(0) + Zort(1) * Zort(1) + Zort(2) * Zort(2));
		Eigen::Matrix3d R;
		R << Xort(0) / Xnorm, Yort(0) / Ynorm, Zort(0) / Znorm,
			Xort(1) / Xnorm, Yort(1) / Ynorm, Zort(1) / Znorm,
			Xort(2) / Xnorm, Yort(2) / Ynorm, Zort(2) / Znorm;
		return R;
	}

	const Eigen::Vector4d LinearAlgebra::normalizeQuatToRot(const Eigen::Vector4d& q)
	{
		double norm = q.norm();
		return (q / norm);
	}

	const Eigen::Matrix4d assembleRotoTranslation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
	{
		Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
		M(3, 3) = 1.0;
		M.topLeftCorner(3, 3) = R;
		M.topRightCorner(3, 1) = t;
		return M;
	}

	const Eigen::Vector4d LinearAlgebra::quaternionInverse(const Eigen::Vector4d& q)
	{
		return (Eigen::Vector4d(q(0), -q(1), -q(2), -q(3)) / q.norm()); // normalized, for rotation computations
	}

	const Eigen::Vector4d LinearAlgebra::quaternionProduct(const Eigen::Vector4d& q, const Eigen::Vector4d& r)
	{
		Eigen::Vector4d qn = normalizeQuatToRot(q);
		Eigen::Vector4d rn = normalizeQuatToRot(r);
		return Eigen::Vector4d(
			rn(0) * qn(0) - rn(1) * qn(1) - rn(2) * qn(2) - rn(3) * qn(3),
			rn(0) * qn(1) + rn(1) * qn(0) - rn(2) * qn(3) + rn(3) * qn(2),
			rn(0) * qn(2) + rn(1) * qn(3) + rn(2) * qn(0) - rn(3) * qn(1),
			rn(0) * qn(3) - rn(1) * qn(2) + rn(2) * qn(1) + rn(3) * qn(0));
	}

	cdouble rSquared(const std::vector<double>& y, const std::vector<double>& f)
	{
		if (y.size() == 0 || f.size() == 0)
			return 0.0;

		const size_t N = y.size();
		if (f.size() != N)
			throw std::invalid_argument("fitting data analysis impossible, dimension mismatch between inputs");

		double SStot = sumSquares<double>(addition<double>(y, -mean<double>(y)));
		double SSres = sumSquares<double>(subtraction<double>(y, f));

		return (1.0 - SSres / SStot);
	}

	RotationSequence::RotationSequence(Axis isQuaternion)
	{
		_isQuat = true;
	}

	RotationSequence::RotationSequence(const Axis first, const Axis second, const Axis third)
	{
		_first = first;
		_second = second;
		_third = third;

		_firstInt = static_cast<uint>(_first) + 1;
		_secondInt = static_cast<uint>(_second) + 1;
		_thirdInt = static_cast<uint>(_third) + 1;
	}

	RotationSequence::RotationSequence(cuint first, cuint second, cuint third)
	{
		if (first != 1 && first != 2 && first != 3)
			throw std::invalid_argument("rotation axis out of range");
		if (second != 1 && second != 2 && second != 3)
			throw std::invalid_argument("rotation axis out of range");
		if (third != 1 && third != 2 && third != 3)
			throw std::invalid_argument("rotation axis out of range");

		_firstInt = first;
		_secondInt = second;
		_thirdInt = third;

		_first = static_cast<Axis>(_firstInt - 1);
		_second = static_cast<Axis>(_secondInt - 1);
		_third = static_cast<Axis>(_thirdInt - 1);
	}

	RotationSequence::RotationSequence(const RotationSequence& seq)
	{
		_first = seq._first;
		_second = seq._second;
		_third = seq._third;

		_firstInt = seq._firstInt;
		_secondInt = seq._secondInt;
		_thirdInt = seq._thirdInt;

		_isQuat = seq._isQuat;
	}

	RotationSequence::RotationSequence(const RotationSequence* seq)
	{
		_first = seq->_first;
		_second = seq->_second;
		_third = seq->_third;

		_firstInt = seq->_firstInt;
		_secondInt = seq->_secondInt;
		_thirdInt = seq->_thirdInt;

		_isQuat = seq->_isQuat;
	}

	RotationSequence& RotationSequence::operator=(const RotationSequence& seq)
	{
		if (&seq == this)
			return *this;

		_first = seq._first;
		_second = seq._second;
		_third = seq._third;

		_firstInt = seq._firstInt;
		_secondInt = seq._secondInt;
		_thirdInt = seq._thirdInt;

		_isQuat = seq._isQuat;

		return *this;
	}

	bool operator!=(const RotationSequence& seq1, const RotationSequence& seq2)
	{
		if (seq1._firstInt != seq2._firstInt || seq1._secondInt != seq2._secondInt || seq1._thirdInt != seq2._thirdInt)
			return true;
		else
			return false;
	}

	const std::string RotationSequence::firstToStr() const
	{
		switch (_first)
		{
		case Axis::X:
			return "rX";
		case Axis::Y:
			return "rY";
		case Axis::Z:
			return "rZ";
		default:
			throw std::invalid_argument("rotation axis outside range");
		}
	}

	const std::string RotationSequence::secondToStr() const
	{
		switch (_second)
		{
		case Axis::X:
			return "rX";
		case Axis::Y:
			return "rY";
		case Axis::Z:
			return "rZ";
		default:
			throw std::invalid_argument("rotation axis outside range");
		}
	}

	const std::string RotationSequence::thirdToStr() const
	{
		switch (_third)
		{
		case Axis::X:
			return "rX";
		case Axis::Y:
			return "rY";
		case Axis::Z:
			return "rZ";
		default:
			throw std::invalid_argument("rotation axis outside range");
		}
	}

	Axis RotationSequence::getNextAxis()
	{
		_cntGet++;
		switch ((_cntGet - 1) % 3)
		{
		case 0:
			return (_first);
		case 1:
			return (_second);
		case 2:
			return (_third);
		}
		throw std::invalid_argument("something went wrong...");
	}

	const cuint RotationSequence::getNextAxisInt()
	{
		_cntGet++;
		switch ((_cntGet - 1) % 3)
		{
		case 0:
			return (_firstInt);
		case 1:
			return (_secondInt);
		case 2:
			return (_thirdInt);
		}
		throw std::invalid_argument("something went wrong...");
	}

	void RotationSequence::setFirst(const Axis axis)
	{
		_first = axis;
		_firstInt = static_cast<uint>(_first) + 1;

		if (_isQuat)
			_isQuat = false;
	}

	void RotationSequence::setFirst(cuint axis)
	{
		if (axis != 1 && axis != 2 && axis != 3)
			throw std::invalid_argument("rotation axis out of range");

		_firstInt = axis;
		_first = static_cast<Axis>(_firstInt - 1);

		if (_isQuat)
			_isQuat = false;
	}

	void RotationSequence::setSecond(const Axis axis)
	{
		_second = axis;
		_secondInt = static_cast<uint>(_second) + 1;

		if (_isQuat)
			_isQuat = false;
	}

	void RotationSequence::setSecond(cuint axis)
	{
		if (axis != 1 && axis != 2 && axis != 3)
			throw std::invalid_argument("rotation axis out of range");

		_secondInt = axis;
		_second = static_cast<Axis>(_secondInt - 1);

		if (_isQuat)
			_isQuat = false;
	}

	void RotationSequence::setThird(const Axis axis)
	{
		_third = axis;
		_thirdInt = static_cast<uint>(_third) + 1;

		if (_isQuat)
			_isQuat = false;
	}

	void RotationSequence::setThird(cuint axis)
	{
		if (axis != 1 && axis != 2 && axis != 3)
			throw std::invalid_argument("rotation axis out of range");

		_thirdInt = axis;
		_third = static_cast<Axis>(_thirdInt - 1);

		if (_isQuat)
			_isQuat = false;
	}

	void RotationSequence::setNextAxis(const Axis axis)
	{
		switch (_cntSet % 3)
		{
		case 0:
			setFirst(axis);
			break;
		case 1:
			setSecond(axis);
			break;
		case 2:
			setThird(axis);
			break;
		}
		_cntSet++;
	}

	void RotationSequence::setNextAxis(cuint axis)
	{
		if (axis != 1 && axis != 2 && axis != 3)
			throw std::invalid_argument("rotation axis out of range");

		setNextAxis(static_cast<Axis>(axis - 1));
	}

	const std::string LinearAlgebra::RotationSequence::toStr() const
	{
		if (_isQuat)
			return std::string("Quaternions");
		else
		{
			std::string s = "";
			for (int i = 0; i < 3; ++i)
			{
				switch (sequence()[i])
				{
				case LinearAlgebra::X:
					s += "X";
					break;
				case LinearAlgebra::Y:
					s += "Y";
					break;
				case LinearAlgebra::Z:
					s += "Z";
					break;
				case LinearAlgebra::Quaternions:
					return std::string("Quaternions");
				default:
					throw std::invalid_argument("Rotation axis out of range.");
				}
			}
			return s;
		}
	}

	const RotationSequence LinearAlgebra::RotationSequence::strToSequence(const std::string& text, bool* ok)
	{
		std::string s;
		for (std::string::const_iterator it = text.cbegin(); it != text.cend(); ++it)
			if (isalpha(*it) != 0)
				s.push_back(*it);

		if (s.size() == 3)
		{
			RotationSequence seq;
			for (std::string::iterator it = s.begin(); it != s.end(); ++it)
			{
				if (*it == 'X' || *it == 'x')
					seq.setNextAxis(Axis::X);
				else if (*it == 'Y' || *it == 'y')
					seq.setNextAxis(Axis::Y);
				else if (*it == 'Z' || *it == 'z')
					seq.setNextAxis(Axis::Z);
			}
			if (ok != nullptr)
				*ok = true;
			return seq;
		}
		else
		{
			if (s == "quaternions" || s == "Quaternions" || s == "quaternion" || s == "Quaternion")
			{
				if (ok != nullptr)
					*ok = true;
				return RotationSequence(Axis::Quaternions);
			}
			else
			{
				if (ok != nullptr)
					*ok = false;
				return RotationSequence(Axis::X, Axis::X, Axis::X);
			}
		}
	}

	const Eigen::Vector3d eulToEul(const RotationSequence srcSeq, const RotationSequence dstSeq, const Eigen::Vector3d& eul)
	{
		return rotmatrixToEul(dstSeq, eulToRotmatrix(srcSeq, eul));
	}

	const Eigen::Matrix3d eulToRotmatrix(Axis first, Axis second, Axis third, const Eigen::Vector3d& eul)
	{
		std::vector<Axis> seq{ first, second, third };
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
		for (int i = 0; i < 3; ++i)
		{
			switch (seq[i])
			{
			case Axis::X:
				R *= rotmatrixX(eul(i));
				break;
			case Axis::Y:
				R *= rotmatrixY(eul(i));
				break;
			case Axis::Z:
				R *= rotmatrixZ(eul(i));
				break;
			default:
				throw std::invalid_argument("euler angles to rotation matrix computation impossible, sequence outside range");
			}
		}
		return R;
	}

	const Eigen::Matrix3d eulToRotmatrix(const RotationSequence seq, const Eigen::Vector3d& eul)
	{
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
		for (int i = 0; i < 3; ++i)
		{
			switch (seq.sequence()[i])
			{
			case Axis::X:
				R *= rotmatrixX(eul(i));
				break;
			case Axis::Y:
				R *= rotmatrixY(eul(i));
				break;
			case Axis::Z:
				R *= rotmatrixZ(eul(i));
				break;
			default:
				throw std::invalid_argument("euler angles to rotation matrix computation impossible, sequence outside range");
			}
		}
		return R;
	}

	const Eigen::Vector4d eulToQuaternion(Axis first, Axis second, Axis third, const Eigen::Vector3d& eul)
	{
		return eulToQuaternion(RotationSequence(first, second, third), eul);
	}

	const Eigen::Vector4d eulToQuaternion(const RotationSequence seq, const Eigen::Vector3d& eul)
	{
		Eigen::Vector4d q(1.0, 0.0, 0.0, 0.0);
		for (int i = 0; i < 3; ++i)
		{
			switch (seq.sequence()[i])
			{
			case Axis::X:
				q = quaternionProduct(q, Eigen::Vector4d(cos(eul(i) / 2), sin(eul(i) / 2), 0.0, 0.0));
				break;
			case Axis::Y:
				q = quaternionProduct(q, Eigen::Vector4d(cos(eul(i) / 2), 0.0, sin(eul(i) / 2), 0.0));
				break;
			case Axis::Z:
				q = quaternionProduct(q, Eigen::Vector4d(cos(eul(i) / 2), 0.0, 0.0, sin(eul(i) / 2)));
				break;
			default:
				throw std::invalid_argument("euler angles to quaternion computation impossible, sequence outside range");
			}
		}
		return (q / q.norm());
	}

	const Eigen::Matrix3d rotmatrix(Axis axis, cdouble a)
	{
		switch (axis)
		{
		case Axis::X:
			return rotmatrixX(a);
		case Axis::Y:
			return rotmatrixY(a);
		case Axis::Z:
			return rotmatrixZ(a);
		default:
			throw std::invalid_argument("rotation matrix computation impossible, axis value out of range");
		}
	}

	const Eigen::Matrix3d rotmatrixX(cdouble a)
	{
		Eigen::Matrix3d R;
		R << 1, 0, 0, 0, cos(a), -sin(a), 0, sin(a), cos(a);
		return R;
	}

	const Eigen::Matrix3d rotmatrixY(cdouble a)
	{
		Eigen::Matrix3d R;
		R << cos(a), 0, sin(a), 0, 1, 0, -sin(a), 0, cos(a);
		return R;
	}

	const Eigen::Matrix3d rotmatrixZ(cdouble a)
	{
		Eigen::Matrix3d R;
		R << cos(a), -sin(a), 0, sin(a), cos(a), 0, 0, 0, 1;
		return R;
	}

	const Eigen::Vector3d rotmatrixToEul(Axis first, Axis second, Axis third, const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);

		if (first == Axis::X && second == Axis::Y && third == Axis::Z)
			return rotmatrixToEulXYZ(R);
		if (first == Axis::X && second == Axis::Z && third == Axis::Y)
			return rotmatrixToEulXZY(R);
		if (first == Axis::Y && second == Axis::X && third == Axis::Z)
			return rotmatrixToEulYXZ(R);
		if (first == Axis::Y && second == Axis::Z && third == Axis::X)
			return rotmatrixToEulYZX(R);
		if (first == Axis::Z && second == Axis::X && third == Axis::Y)
			return rotmatrixToEulZXY(R);
		if (first == Axis::Z && second == Axis::Y && third == Axis::X)
			return rotmatrixToEulZYX(R);
		if (first == Axis::X && second == Axis::Y && third == Axis::X)
			return rotmatrixToEulXYX(R);
		if (first == Axis::X && second == Axis::Z && third == Axis::X)
			return rotmatrixToEulXZX(R);
		if (first == Axis::Y && second == Axis::X && third == Axis::Y)
			return rotmatrixToEulYXY(R);
		if (first == Axis::Y && second == Axis::Z && third == Axis::Y)
			return rotmatrixToEulYZY(R);
		if (first == Axis::Z && second == Axis::X && third == Axis::Z)
			return rotmatrixToEulZXZ(R);
		if (first == Axis::Z && second == Axis::Y && third == Axis::Z)
			return rotmatrixToEulZYZ(R);
		throw std::invalid_argument("rotation matrix to Euler angles impossible, combination is unfeasible");
	}

	const Eigen::Vector3d rotmatrixToEul(const RotationSequence seq, const Eigen::Matrix3d& R)
	{
		if (seq == RotationSequence(Axis::X, Axis::Y, Axis::Z))
			return rotmatrixToEulXYZ(R);
		if (seq == RotationSequence(Axis::X, Axis::Z, Axis::Y))
			return rotmatrixToEulXZY(R);
		if (seq == RotationSequence(Axis::Y, Axis::X, Axis::Z))
			return rotmatrixToEulYXZ(R);
		if (seq == RotationSequence(Axis::Y, Axis::Z, Axis::X))
			return rotmatrixToEulYZX(R);
		if (seq == RotationSequence(Axis::Z, Axis::X, Axis::Y))
			return rotmatrixToEulZXY(R);
		if (seq == RotationSequence(Axis::Z, Axis::Y, Axis::X))
			return rotmatrixToEulZYX(R);
		if (seq == RotationSequence(Axis::X, Axis::Y, Axis::X))
			return rotmatrixToEulXYX(R);
		if (seq == RotationSequence(Axis::X, Axis::Z, Axis::X))
			return rotmatrixToEulXZX(R);
		if (seq == RotationSequence(Axis::Y, Axis::X, Axis::Y))
			return rotmatrixToEulYXY(R);
		if (seq == RotationSequence(Axis::Y, Axis::Z, Axis::Y))
			return rotmatrixToEulYZY(R);
		if (seq == RotationSequence(Axis::Z, Axis::X, Axis::Z))
			return rotmatrixToEulZXZ(R);
		if (seq == RotationSequence(Axis::Z, Axis::Y, Axis::Z))
			return rotmatrixToEulZYZ(R);
		throw std::invalid_argument("rotation matrix to Euler angles impossible, combination is unfeasible");
	}

	const Eigen::Vector3d rotmatrixToEulXYZ(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		return Eigen::Vector3d(atan2(-R(1, 2), R(2, 2)),
			atan2(R(0, 2), sqrt(R(1, 2) * R(1, 2) + R(2, 2) * R(2, 2))),
			atan2(-R(0, 1), R(0, 0)));
	}

	const Eigen::Vector3d rotmatrixToEulXZY(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		return Eigen::Vector3d(atan2(R(2, 1), R(1, 1)),
			atan2(-R(0, 1), sqrt(R(0, 0) * R(0, 0) + R(0, 2) * R(0, 2))),
			atan2(R(0, 2), R(0, 0)));
	}

	const Eigen::Vector3d rotmatrixToEulYXZ(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		return Eigen::Vector3d(atan2(R(0, 2), R(2, 2)),
			atan2(-R(1, 2), sqrt(R(0, 2) * R(0, 2) + R(2, 2) * R(2, 2))),
			atan2(R(1, 0), R(1, 1)));
	}

	const Eigen::Vector3d rotmatrixToEulYZX(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		return Eigen::Vector3d(atan2(-R(2, 0), R(0, 0)),
			atan2(R(1, 0), sqrt(R(0, 0) * R(0, 0) + R(2, 0) * R(2, 0))),
			atan2(-R(1, 2), R(1, 1)));
	}

	const Eigen::Vector3d rotmatrixToEulZXY(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		return Eigen::Vector3d(atan2(-R(0, 1), R(1, 1)),
			atan2(R(2, 1), sqrt(R(2, 0) * R(2, 0) + R(2, 2) * R(2, 2))),
			atan2(-R(2, 0), R(2, 2)));
	}

	const Eigen::Vector3d rotmatrixToEulZYX(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		return Eigen::Vector3d(atan2(R(1, 0), R(0, 0)),
			atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2))),
			atan2(R(2, 1), R(2, 2)));
	}

	const Eigen::Vector3d rotmatrixToEulXYX(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		if (R(0, 0) < 1)
			return Eigen::Vector3d(atan2(R(1, 0), -R(2, 0)),
				atan2(sqrt(R(0, 1) * R(0, 1) + R(0, 2) * R(0, 2)), R(0, 0)),
				atan2(R(0, 1), R(0, 2)));
		else
			return Eigen::Vector3d(atan2(-R(1, 2), R(1, 1)), 0, 0);
	}

	const Eigen::Vector3d rotmatrixToEulXZX(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		if (R(0, 0) < 1)
			return Eigen::Vector3d(atan2(R(2, 0), R(1, 0)),
				atan2(sqrt(R(1, 0) * R(1, 0) + R(2, 0) * R(2, 0)), R(0, 0)),
				atan2(R(0, 2), -R(0, 1)));
		else
			return Eigen::Vector3d(atan2(R(2, 1), R(2, 2)), 0, 0);
	}

	const Eigen::Vector3d rotmatrixToEulYXY(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		if (R(1, 1) < 1)
			return Eigen::Vector3d(atan2(R(0, 1), R(2, 1)),
				atan2(sqrt(R(1, 0) * R(1, 0) + R(1, 2) * R(1, 2)), R(1, 1)),
				atan2(R(1, 0), -R(1, 2)));
		else
			return Eigen::Vector3d(atan2(R(0, 2), R(0, 0)), 0, 0);
	}

	const Eigen::Vector3d rotmatrixToEulYZY(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		if (R(1, 1) < 1)
			return Eigen::Vector3d(atan2(R(2, 1), -R(0, 1)),
				atan2(sqrt(R(1, 0) * R(1, 0) + R(1, 2) * R(1, 2)), R(1, 1)),
				atan2(R(1, 2), R(1, 0)));
		else
			return Eigen::Vector3d(atan2(-R(2, 0), R(2, 2)), 0, 0);
	}

	const Eigen::Vector3d rotmatrixToEulZXZ(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		if (R(2, 2) < 1)
			return Eigen::Vector3d(atan2(R(0, 2), -R(1, 2)),
				atan2(sqrt(R(0, 2) * R(0, 2) + R(1, 2) * R(1, 2)), R(2, 2)),
				atan2(R(2, 0), R(2, 1)));
		else
			return Eigen::Vector3d(atan2(-R(0, 1), R(0, 0)), 0, 0);
	}

	const Eigen::Vector3d rotmatrixToEulZYZ(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		if (R(2, 2) < 1.0)
			return Eigen::Vector3d(atan2(R(1, 2), R(0, 2)),
				atan2(sqrt(R(2, 0) * R(2, 0) + R(2, 1) * R(2, 1)), R(2, 2)),
				atan2(R(2, 1), -R(2, 0)));
		else
			return Eigen::Vector3d(atan2(R(1, 0), R(1, 1)), 0, 0);
	}

	const Eigen::Vector4d rotmatrixToQuaternion(const Eigen::Matrix3d& M)
	{
		Eigen::Matrix3d R = normalizeMatrixToRot(M);
		Eigen::Vector4d q;

		double r = sqrt(1 + R.trace());
		q(0) = 0.5 * r;
		q(1) = copysign(0.5 * sqrt(roundoff(1 + R(0, 0) - R(1, 1) - R(2, 2), 20)), R(2, 1) - R(1, 2));
		q(2) = copysign(0.5 * sqrt(roundoff(1 - R(0, 0) + R(1, 1) - R(2, 2), 20)), R(0, 2) - R(2, 0));
		q(3) = copysign(0.5 * sqrt(roundoff(1 - R(0, 0) - R(1, 1) + R(2, 2), 20)), R(1, 0) - R(0, 1));

		return q;
	}

	const Eigen::Vector3d quaternionToEul(Axis first, Axis second, Axis third, const Eigen::Vector4d& q)
	{
		return quaternionToEul(RotationSequence(first, second, third), q);
	}

	const Eigen::Vector3d quaternionToEul(const RotationSequence seq, const Eigen::Vector4d& q)
	{
		return rotmatrixToEul(seq, quaternionToRotmatrix(normalizeQuatToRot(q)));
	}

	const Eigen::Matrix3d quaternionToRotmatrix(const Eigen::Vector4d& p)
	{
		Eigen::Vector4d q = normalizeQuatToRot(p);

		Eigen::Matrix3d R;
		double s = (q(0) * q(0) + q(1) * q(1) + q(2) * q(2) + q(3) * q(3));
		R(0, 0) = 1.0 - 2.0 * (q(2) * q(2) + q(3) * q(3)) / s;
		R(0, 1) = 2.0 * (q(1) * q(2) - q(3) * q(0)) / s;
		R(0, 2) = 2.0 * (q(1) * q(3) + q(2) * q(0)) / s;
		R(1, 0) = 2.0 * (q(1) * q(2) + q(3) * q(0)) / s;
		R(1, 1) = 1.0 - 2.0 * (q(1) * q(1) + q(3) * q(3)) / s;
		R(1, 2) = 2.0 * (q(2) * q(3) - q(1) * q(0)) / s;
		R(2, 0) = 2.0 * (q(1) * q(3) - q(2) * q(0)) / s;
		R(2, 1) = 2.0 * (q(3) * q(2) + q(1) * q(0)) / s;
		R(2, 2) = 1.0 - 2.0 * (q(2) * q(2) + q(1) * q(1)) / s;
		return R;
	}

	namespace CellAlignment
	{
		const CellTransformation cellAlignmentAlgorithm(const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& r, cuint steadyStateN, cdouble steadyStateThres, cuint maxIter, const LinearSolver solver)
		{
			// validate if input measurements have the same dimension
			const size_t N = l.size();
			if (p.size() != N || r.size() != N)
				throw std::invalid_argument("rototranslation estimate impossible, measurements dimension mismatch");

			// pre-compute rotation matrices associated to each robot pose orientation r (zyx)
			std::vector<Eigen::Matrix3d> R(N);
			for (size_t i = 0; i < N; ++i)
				R[i] = eulToRotmatrix(Axis::Z, Axis::Y, Axis::X, r[i]);

			// initialize estimate for TCP
			Eigen::Vector3d v(1, 1, 1);

			// initialize estimate for M
			const Eigen::Vector3d lmean = sum(l) / N;
			Eigen::MatrixX3d Ytranspose(N, 3);
			for (size_t i = 0; i < N; ++i)
				Ytranspose.row(i) = (l[i] - lmean).transpose();
			Eigen::Matrix4d M = estimateRotoTranslation(lmean, Ytranspose, p, R, v);

			// start iterations
			uint steadyStateCnt = 0;
			for (uint iter = 0; iter < maxIter && steadyStateCnt < steadyStateN; ++iter)
			{
				// save old values
				Eigen::Vector3d vold = v;
				Eigen::Matrix4d Mold = M;

				// update estimates
				v = estimateTCPTranslation(l, p, R, M, solver);
				M = estimateRotoTranslation(lmean, Ytranspose, p, R, v);

				// check for results steady-state
				if (checkSteadyState(Mold, M, vold, v, steadyStateThres))
					++steadyStateCnt;
				else
					steadyStateCnt = 0;
			}

			// output estimates
			CellTransformation res;
			res.M = M;
			res.TCP = v;
			return res;
		}

		const Eigen::Matrix4d estimateRotoTranslation(const Eigen::Vector3d& lmean, const Eigen::MatrixX3d& Ytranspose, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Matrix3d>& R, const Eigen::Vector3d& v)
		{
			const size_t N = p.size();

			// calculate q and its centroid, then build matrix X
			std::vector<Eigen::Vector3d> q(N);
			for (size_t i = 0; i < N; ++i)
				q[i] = p[i] + R[i] * v;
			const Eigen::Vector3d qmean = sum(q) / N;
			Eigen::Matrix3Xd X(3, N);
			for (size_t i = 0; i < N; ++i)
				X.col(i) = q[i] - qmean;

			// compute covariance matrix and its SVD
			const Eigen::JacobiSVD<Eigen::Matrix3d> svd(X * Ytranspose, Eigen::ComputeFullU | Eigen::ComputeFullV);
			const Eigen::Matrix3d U = svd.matrixU();
			const Eigen::Matrix3d V = svd.matrixV();

			// compute rotation estimate
			const Eigen::Matrix3d RM = V * Eigen::Matrix3d(Eigen::Vector3d(1.0, 1.0, (V * U.transpose()).determinant()).asDiagonal()) * U.transpose();

			// compute translation estimate
			const Eigen::Vector3d tM = lmean - RM * qmean;

			// return the rototranslation matrix
			return assembleRotoTranslation(RM, tM);
		}

		const Eigen::Vector3d estimateTCPTranslation(const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Matrix3d>& R, const Eigen::Matrix4d& M, const LinearSolver solver)
		{
			const size_t N = p.size();

			// extract rotation and translation from matrix M
			const Eigen::Matrix3d RM = M.block(0, 0, 3, 3);
			const Eigen::Vector3d tM = M.block(0, 3, 3, 1);

			// build variables for least-square optimization
			Eigen::MatrixX3d A(3 * N, 3);
			Eigen::VectorXd b(3 * N);
			for (size_t i = 0; i < N; ++i)
			{
				A.block(3 * i, 0, 3, 3) = RM * R[i];
				b.segment(3 * i, 3) = l[i] - RM * p[i] - tM;
			}

			// return the estimate of tcp
			switch (solver)
			{
				case LinearAlgebra::HouseHolderQR:
					return A.householderQr().solve(b);
				case LinearAlgebra::ColPivHouseHolderQR:
					return A.colPivHouseholderQr().solve(b);
				case LinearAlgebra::FullPivHouseHolderQR:
					return A.fullPivHouseholderQr().solve(b);
				case LinearAlgebra::CholeskyDec:
					return (A.transpose() * A).ldlt().solve(A.transpose() * b);
				case LinearAlgebra::Inversion:
				{
					Eigen::Matrix3d R;
					bool flag;
					(A.transpose() * A).computeInverseWithCheck(R, flag);
					if (!flag)
						throw std::invalid_argument("matrix inversion failed");
					return R * A.transpose() * b;
				}
				default:
				case LinearAlgebra::SVD:
					return A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
				}
		}

		const bool checkSteadyState(const Eigen::Matrix4d& Mold, const Eigen::Matrix4d& M, const Eigen::Vector3d& vold, const Eigen::Vector3d& v, cdouble threshold)
		{
			for (int i = 0; i < 4; ++i)
			{
				for (int j = 0; j < 4; ++j)
					if (abs(M(i, j) - Mold(i, j)) > (threshold * abs(Mold(i, j))))
						return false;
				if (i < 3 && (abs(v(i) - vold(i)) > (threshold * abs(vold(i)))))
					return false;
			}
			return true;
		}

		const std::vector<Eigen::Vector3d> evaluateModel(const Eigen::Matrix4d& M, const Eigen::Vector3d& v, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& r)
		{
			// evalute the model Li = M * (Pi + Ri * TCP), i.e., computes L from the given arguments

			// validate if input measurements have the same dimension
			const size_t N = p.size();
			if (r.size() != N)
				throw std::invalid_argument("model evaluation impossible, measurements dimension mismatch");

			// calculate each i-th component of the laser measurements
			std::vector<Eigen::Vector3d> l(N);
			for (size_t i = 0; i < N; ++i)
			{
				Eigen::Vector4d tmp;
				tmp << p[i] + (eulToRotmatrix(Axis::Z, Axis::Y, Axis::X, r[i]) * v), 1;
				l[i] = (M * tmp).head(3);
			}
			return l;
		}

		const double LinearAlgebra::assessModel(const std::vector<Eigen::Vector3d>& l_fitted, const std::vector<Eigen::Vector3d>& l)
		{
			// validate if input measurements have the same dimension
			const size_t N = l.size();
			if (l_fitted.size() != N)
				throw std::invalid_argument("model evaluation impossible, measurements dimension mismatch");

			// evaluate model and divide result in each dimension
			std::vector<double> l_fitted_x(N);
			std::vector<double> l_fitted_y(N);
			std::vector<double> l_fitted_z(N);
			std::vector<double> l_x(N);
			std::vector<double> l_y(N);
			std::vector<double> l_z(N);
			std::vector<double> l_fitted_components[3];
			std::vector<double> l_components[3];
			for (size_t i = 0; i < N; ++i)
			{
				l_fitted_x[i] = l_fitted[i].x();
				l_fitted_y[i] = l_fitted[i].y();
				l_fitted_z[i] = l_fitted[i].z();
				l_x[i] = l[i].x();
				l_y[i] = l[i].y();
				l_z[i] = l[i].z();
			}

			// assess each dimension
			double R2_x = rSquared(l_x, l_fitted_x);
			double R2_y = rSquared(l_y, l_fitted_y);
			double R2_z = rSquared(l_z, l_fitted_z);

			return ((R2_x + R2_y + R2_z) / 3.0);
		}

		const std::vector<std::pair<unsigned int, double>> getWorstFittingData(const std::vector<Eigen::Vector3d>& l_fitted, const std::vector<Eigen::Vector3d>& l_true, cdouble threshold, cuint N)
		{
			// validate if input measurements have the same dimension
			const size_t L = l_true.size();
			if (l_fitted.size() != L)
				throw std::invalid_argument("fitting data analysis impossible, measurements dimension mismatch");
			
			// calculate norm of the i-th difference and save it only if > threshold
			std::vector<std::pair<unsigned int, double>> norms;
			for (size_t i = 0; i < L; ++i)
			{
				double norm = (l_true[i] - l_fitted[i]).norm();
				if (norm >= threshold)
					norms.push_back(std::make_pair((unsigned int)i, norm));
			}
			
			// partially sort first n elements of norms vector
			if (norms.empty())
				return std::vector<std::pair<unsigned int, double>>{ };
			const unsigned int n = (N > norms.size()) ? norms.size() : N;
			std::partial_sort(norms.begin(), norms.begin() + n, norms.end(), [](const std::pair<unsigned int, double> &i, const std::pair<unsigned int, double> &j) { return i.second > j.second; });

			// return indeces, from largest to smallest norm
			std::vector<std::pair<unsigned int, double>> indeces(n);
			for (unsigned int i = 0; i < n; ++i)
				indeces[i] = norms[i];
			return indeces;
		}
	}
}