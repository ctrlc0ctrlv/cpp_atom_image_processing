#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <eigen3/Eigen/SVD>
#include "delyana/cv.h"
#include "cpd.h"

using namespace delyana;

void test_pic()
{
	// preparation
	Image img;
	img.load("finalimage9.png");

	// main cycle
	std::chrono::high_resolution_clock clock;
	auto time = clock.now();
	auto contours = findContoursMut(img.dilate(3).binarize());
	std::chrono::duration<double> d = clock.now() - time;

	// output
	std::cout << "Execution time: " << d.count() << " seconds" << std::endl;
	std::cout << "Number of contours:" << std::endl;
	std::cout << contours.size() << std::endl;
	for (auto p : contours)
		std::cout << "(" << p.x << ", " << p.y << ", " << p.dx << ", " << p.dy << ")";
	std::cout << std::endl;
	img.save("after.png");

	Eigen::MatrixX2f X(contours.size(), 2);

	int M = 64;
	Eigen::MatrixX2f Y(M, 2);

	for (int i = 0; i < contours.size(); i++)
	{
		X(i, 0) = contours[i].x;
		X(i, 1) = contours[i].y;
	}

	for (int i = 0; i < M; i++)
	{
		Y(i, 0) = (i / 8) * 10;
		Y(i, 1) = (i % 8) * 10;
	}

	AffineTransform transform;
	CPD cpd;
	time = clock.now();
	cpd(X, Y, transform);
	d = clock.now() - time;
	std::cout << "Execution time: " << d.count() << " seconds" << std::endl;

	Eigen::Matrix2f R = transform.linear();
	Eigen::Vector2f tr = transform.translation();
	std::cout << R << "\n"
			  << tr << std::endl;
	// Y = (Y * R).colwise() + tr;
	for (int i = 0; i < M; i++)
	{
		std::cout << "(" << Y(i, 0) << ", " << Y(i, 1) << "), ";
	}
	std::cout << std::endl;
	Y = (Y * R.transpose()).rowwise() + tr.transpose();
	for (int i = 0; i < M; i++)
	{
		std::cout << "(" << Y(i, 0) << ", " << Y(i, 1) << "), ";
	}
	std::cout << std::endl;

	//
	// big test!
	//
	d *= 0;
	int cnt = 0;
	float total = 0;
	std::string path = "./pngs/content/pngs/";
	for (const auto &entry : std::filesystem::directory_iterator(path))
	{
		img.load(entry.path());
		time = clock.now();
		contours = findContoursMut(img.dilate(3).binarize());

		d = clock.now() - time;
		total += d.count();

		Eigen::MatrixX2f X(contours.size(), 2);
		for (int i = 0; i < contours.size(); i++)
		{
			X(i, 0) = contours[i].x;
			X(i, 1) = contours[i].y;
		}
		time = clock.now();
		// cpd.w = 0.0f;
		cpd(X, Y, transform);

		std::chrono::duration<double> d2 = clock.now() - time;
		total += d2.count();

		// Eigen::Matrix2f R = transform.linear();
		// Eigen::Vector2f tr = transform.translation();
		// std::cout << R << "\n" << tr << std::endl;
		//std::cout << d.count() * 1000 << "\t" << d2.count() * 1000 << std::endl;
		// std::cout << entry.path() << std::endl;
		cnt++;

		Eigen::MatrixXf P(contours.size(), M);
		cpd.getP(P);
		Eigen::Matrix2f R = transform.linear();
		Eigen::Vector2f tr = transform.translation();
		Eigen::MatrixX2f Y_out = (Y * R.transpose()).rowwise() + tr.transpose();
		//std::cout << R << "\n"
		//		  << tr << std::endl;
		if (cnt == 76){
			std::cout << P << std::endl;
			for (int i = 0; i < contours.size(); i++)
			{
				std::cout << "(" << X(i, 0) << ", " << X(i, 1) << "), ";
			}
			std::cout << std::endl;
			for (int i = 0; i < M; i++)
			{
				std::cout << "(" << Y(i, 0) << ", " << Y(i, 1) << "), ";
			}
			std::cout << std::endl;
			for (int i = 0; i < M; i++)
			{
				std::cout << "(" << Y_out(i, 0) << ", " << Y_out(i, 1) << "), ";
			}
			std::cout << std::endl;
		}

		// calculating MSE
		float sum_sqr = 0;
		int sum_cnt = 0;
		for (int i = 0; i < P.rows(); i++)
		{
			for (int j = 0; j < P.cols(); j++)
			{
				if (P(i, j) > 0.5){
					sum_cnt++;
					sum_sqr += (Y_out(i, 0) - X(j, 0)) * (Y_out(i, 0) - X(j, 0)) + (Y_out(i, 1) - X(j, 1)) * (Y_out(i, 1) - X(j, 1));
				}
			}
		}
		// picture number << MSE << points correlated
		std::cout << cnt << "\t" << sum_sqr / sum_cnt << "\t" << sum_cnt << std::endl;
	}
	// img.load("./pngs/content/pngs/finalimage11.png");
	std::cout << std::endl
			  << "Execution time: " << total << " seconds" << std::endl;
	std::cout << cnt << std::endl;
}

void fileTest(int num)
{
	std::ifstream infile("txts/" + std::to_string(num) + "_new.txt");
	int a, b;
	int cnt = 0;
	while (infile >> a >> b)
	{
		cnt++;
	}
	infile.close();

	Eigen::MatrixX2f X(cnt, 2);

	infile.open("txts/" + std::to_string(num) + "_new.txt");
	cnt = 0;
	while (infile >> a >> b)
	{
		X(cnt, 0) = a;
		X(cnt, 1) = b;
		cnt++;
	}
	infile.close();

	infile.open("txts/" + std::to_string(num) + ".txt");
	int k = 0;
	while (infile >> a >> b)
	{
		k++;
	}
	infile.close();
	Eigen::MatrixX2f Y(k, 2);

	infile.open("txts/" + std::to_string(num) + ".txt");
	k = 0;
	while (infile >> a >> b)
	{
		Y(k, 0) = a;
		Y(k, 1) = b;
		k++;
	}
	infile.close();

	AffineTransform transform;
	CPD cpd;
	std::chrono::high_resolution_clock clock;
	auto time = clock.now();
	cpd(X, Y, transform);
	std::chrono::duration<double> d = clock.now() - time;
	std::cout << "Execution time for N = " << num*num << " M = " << cnt << " : " << d.count() << " seconds" << std::endl;
}

void MTest(int num, int mmax)
{
	std::ifstream infile("txts/" + std::to_string(num) + "_new.txt");
	int a, b;
	int cnt = 0;
	while (infile >> a >> b)
	{
		cnt++;
	}
	infile.close();

	Eigen::MatrixX2f X(cnt, 2);

	infile.open("txts/" + std::to_string(num) + "_new.txt");
	cnt = 0;
	while (infile >> a >> b)
	{
		X(cnt, 0) = a;
		X(cnt, 1) = b;
		cnt++;
	}
	infile.close();

	Eigen::MatrixX2f Y(mmax*mmax, 2);

	infile.open("txts/" + std::string("75") + ".txt");
	int k = 0;
	while (k < mmax * mmax)
	{
		infile >> a >> b;
		Y(k, 0) = a;
		Y(k, 1) = b;
		k++;
	}
	infile.close();

	double sum = 0;
	for (int i = 0; i < 100; i++){
		AffineTransform transform;
		CPD cpd;
		std::chrono::high_resolution_clock clock;
		auto time = clock.now();
		cpd(X, Y, transform);
		std::chrono::duration<double> d = clock.now() - time;
		sum += d.count();
	}
	// std::cout << cnt << "\t" << mmax*mmax << "\t" << sum / 5 << std::endl;
	std::cout << sum / 100 << std::endl;
}

void NTest(int num, int mmax)
{
	std::ifstream infile("txts/" + std::to_string(num) + "_new.txt");
	int a, b;
	int cnt = 0;
	while (infile >> a >> b)
	{
		cnt++;
	}
	infile.close();

	Eigen::MatrixX2f X(cnt, 2);

	infile.open("txts/" + std::to_string(num) + "_new.txt");
	cnt = 0;
	while (infile >> a >> b)
	{
		X(cnt, 0) = a;
		X(cnt, 1) = b;
		cnt++;
	}
	infile.close();

	Eigen::MatrixX2f Y(mmax*mmax, 2);

	infile.open("txts/" + std::string("75") + ".txt");
	int k = 0;
	while (k < mmax * mmax)
	{
		infile >> a >> b;
		Y(k, 0) = a;
		Y(k, 1) = b;
		k++;
	}
	infile.close();

	double sum = 0;
	for (int i = 0; i < 10; i++){
		AffineTransform transform;
		CPD cpd;
		std::chrono::high_resolution_clock clock;
		auto time = clock.now();
		cpd(X, Y, transform);
		std::chrono::duration<double> d = clock.now() - time;
		sum += d.count();
	}
	std::cout << cnt << "\t" << mmax*mmax << "\t" << sum / 10 << std::endl;
	// std::cout << sum / 100 << std::endl;
}

void old_main()
{
	// int x[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 23, 25, 30, 35, 40, 45, 50};
	// int y[] = {2, 5, 10, 20, 30, 50};
	// for (auto i : y){
	//	for (auto j : x)
	//		MTest(i, j);
	// 	std::cout << std::endl;
	// }

	int x[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 23, 25, 30, 35, 40, 45, 50, 55, 60};
	int y[] = {2, 5, 10, 20, 30, 50, 75};
	for (auto i : y){
		for (auto j : x)
			NTest(j, i);
		std::cout << std::endl;
	}
}

void CPD::getP(Eigen::MatrixXf &mtr)
{
	mtr = P;
}

void sampleMSE()
{
	int num = 5;
	std::ifstream infile("txts/" + std::to_string(num) + "_new.txt");
	int a, b;
	int cnt = 0;
	while (infile >> a >> b)
	{
		cnt++;
	}
	infile.close();

	Eigen::MatrixX2f X(cnt, 2);

	infile.open("txts/" + std::to_string(num) + "_new.txt");
	cnt = 0;
	while (infile >> a >> b)
	{
		X(cnt, 0) = a;
		X(cnt, 1) = b;
		cnt++;
	}
	infile.close();

	int M = num*num;
	Eigen::MatrixX2f Y(M, 2);

	infile.open("txts/" + std::to_string(num) + ".txt");
	int k = 0;

	while (k < M)
	{
		infile >> a >> b;
		Y(k, 0) = a + 15;
		Y(k, 1) = b - 15;
		k++;
	}
	infile.close();

	AffineTransform transform;
	CPD cpd;
	cpd(X, Y, transform);

	Eigen::MatrixXf P(cnt, M);
	cpd.getP(P);
	std::cout << P << std::endl;

	Eigen::Matrix2f R = transform.linear();
	Eigen::Vector2f tr = transform.translation();
	std::cout << R << "\n"
			  << tr << std::endl;
	for (int i = 0; i < M; i++)
	{
		std::cout << "(" << Y(i, 0) << ", " << Y(i, 1) << "), ";
	}
	std::cout << std::endl;
	Eigen::MatrixX2f Y_out = (Y * R.transpose()).rowwise() + tr.transpose();
	for (int i = 0; i < M; i++)
	{
		std::cout << "(" << Y_out(i, 0) << ", " << Y_out(i, 1) << "), ";
	}
	std::cout << std::endl;

	// calculating MSE
	float sum_sqr = 0;
	int sum_cnt = 0;
	for (int i = 0; i < P.rows(); i++)
	{
		for (int j = 0; j < P.cols(); j++)
		{
			if (P(i, j) > 0.5){
				sum_cnt++;
				sum_sqr += (Y_out(i, 0) - X(j, 0)) * (Y_out(i, 0) - X(j, 0)) + (Y_out(i, 1) - X(j, 1)) * (Y_out(i, 1) - X(j, 1));
			}
		}
	}
	std::cout << "MSE = " << sum_sqr / sum_cnt << " Points = " << sum_cnt << std::endl;
}

int main()
{
	test_pic();
}
