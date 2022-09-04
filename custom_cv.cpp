#include "custom_cv.h"

#include <vector>

#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png.hpp>

std::ostream &operator<<(std::ostream &os, const CustomPoint &p)
{
	os << "(" << p.x << ", " << p.y << ", " << p.sx << ", " << p.sy << ")";
	return os;
}

bool operator<(const CustomPoint &p1, const CustomPoint &p2) noexcept
{
	if (p1.x != p2.x)
		return p1.x < p2.x;
	else
		return p1.y < p2.y;
}

bool operator==(const CustomPoint &p1, const CustomPoint &p2) noexcept
{
	bool f1 = abs(p1.x - p2.x) == 0;
	bool f2 = abs(p1.y - p2.y) == 0;
	bool f3 = abs(p1.sx - p2.sx) < __FLT_EPSILON__;
	bool f4 = abs(p1.sy - p2.sy) < __FLT_EPSILON__;
	return (f1 && f2 && f3 && f4);
}

float sqrtVariance(float mean, const std::vector<int> &vec)
{
	auto add_square = [mean](float sum, int i)
	{
		auto d = i - mean;
		return sum + d * d;
	};
	float total = std::accumulate(vec.begin(), vec.end(), 0.0, add_square);
	// return sqrt(total);
	return sqrt(total / (vec.size()));
	// return sqrt(total / (vec.size() - 1)); // ???
}
