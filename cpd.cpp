#include "cpd.h"
// #include "logging.h"
#include <cmath>
#include <stdexcept>
#include <eigen3/Eigen/SVD>

using namespace	Eigen;

/**
 * @brief Matrix normalization.
 *
 * @param[in/out] X - matrix to normalize.
 * @param[out] t0 - translation vector to compute.
 * @param[out] stdev - standard deviation to compute.
 *
 * Brings matrix's mean to (0; 0) and dispersion to 1.
 * Also computes translation and scaling of such operation for later usage.
 */
static void normalizeMatrix(Eigen::MatrixX2f &X, Eigen::Vector2f &t0, float &stdev)
{
	// Firstly compute mean (aka translation vector) and dispersion (aka scaling)
	t0 = X.colwise().mean();
	stdev = std::sqrt((X.rowwise() - t0.transpose()).squaredNorm() / X.rows());
	// Then change matrix
	X = (X.rowwise() - t0.transpose()) / stdev;
}

void CPD::init(const MatrixX2f &X, const MatrixX2f &Y)
{
	this->X = X;
	this->Y = Y;
	s = 1.f;
	N = X.rows();
	M = Y.rows();
	P.resize(M, N);
	R = Matrix2f::Identity(); // D x D
	t = Vector2f::Zero(); // D x 1
	normalizeMatrix(this->X, tx, dispX);
	normalizeMatrix(this->Y, ty, dispY);

	sigmaSqr = 0.f;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			sigmaSqr += (X.row(i) - Y.row(j)).squaredNorm();
	sigmaSqr /= N * M * D;
}

void CPD::Estep()
{
	// Numerator part: calculate all Gauss exponents and store them in matrix P
#	pragma omp parallel for num_threads(4)
	for (int m = 0; m < M; m++)
	{
		const Vector2f sRYt = s * R * Y.row(m).transpose() + t; // D x 1
		for (int n = 0; n < N; n++)
			P(m, n) = std::exp((X.row(n) - sRYt.transpose()).squaredNorm() / (-2.f * sigmaSqr));
	}

	// Denominator part
	const float c = std::pow(2.f*M_PI*sigmaSqr, D/2.f) * w * M / ((1.f - w) * N);
#	pragma omp parallel for num_threads(4)
	for (int n = 0; n < N; n++)
	{
		const float denominator = P.col(n).sum() + c;
		P.col(n) /= denominator;
	}
}

void CPD::Mstep(bool changeScaling)
{
	const VectorXf p1 = P.rowwise().sum(); // M x 1
	const VectorXf pt1 = P.colwise().sum().transpose(); // N x 1

	const float Np = p1.sum(); // scalar

	const Vector2f mu_x = (X.transpose() * pt1) / Np; // D x 1
	const Vector2f mu_y = (Y.transpose() * p1) / Np; // D x 1

	const MatrixX2f hatX = X - VectorXf::Ones(N) * mu_x.transpose(); // N x D
	const MatrixX2f hatY = Y - VectorXf::Ones(M) * mu_y.transpose(); // M x D

	const Matrix2f A = hatX.transpose() * P.transpose() * hatY; // DxN * NxM * MxD = DxD

	Eigen::JacobiSVD<Matrix2f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const Matrix2f &U = svd.matrixU(); // D x D
	const Matrix2f VT = svd.matrixV().transpose(); // D x D

	// (!!!) det(U*VT) = sign(det(A)) = +-1
	Matrix2f C = Matrix2f::Identity();
	if (A.determinant() < 0)
		C(1, 1) = -1.f;

	R = U * C * VT; // D x D

	if (changeScaling)
	{
		s = (A.transpose() * R).trace();
		s /= (hatY.transpose() * p1.asDiagonal() * hatY).trace(); // DxM * MxM * MxD = DxD
	}

	t = mu_x - s * R * mu_y;

	sigmaSqr = (hatX.transpose() * pt1.asDiagonal() * hatX).trace();
	sigmaSqr -= s * (A.transpose() * R).trace();
	sigmaSqr /= Np * D;
	sigmaSqr = std::fabs(sigmaSqr);
}

void CPD::operator ()(const MatrixX2f &X, const MatrixX2f &Y, AffineTransform &transform)
{
	if(X.rows() == 0 || X.cols() == 0 || Y.rows() == 0 || Y.cols() == 0)
		throw std::invalid_argument("CPD::operator(): matrices must not be empty");
	init(X, Y);
	int i = 0;
	while(i < maxItersNoScaling && sigmaSqr > 0)
	{
		Estep();
		Mstep(false);
		// LOG_CORE_DEBUG("#{}, no scaling: CPD sigmaSqr = {}", i, sigmaSqr);
		i++;
	}
	i = 0;
	while(i < maxItersWithScaling && sigmaSqr > 0)
	{
		Estep();
		Mstep(true);
		// LOG_CORE_DEBUG("#{}, with scaling: CPD sigmaSqr = {}", i, sigmaSqr);
		i++;
	}
	// Without normalization:
	// transform.linear() = R * s;
	// transform.translation() = this->t;

	// With normalization:
	transform.linear() = R * s * dispX / dispY;
	// (A * B).transpose() = B.transpose() * A.transpose(), that's why
	transform.translation() = -transform.linear() * ty + this->t * dispX + tx;
}
