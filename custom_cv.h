#ifndef CUSTOM_CV_H
#define CUSTOM_CV_H

#include <iostream>
#include <ostream>

/**
 * @brief Simple point structure.
 *
 * @param x - x coordinate.
 * @param y - y coordinate.
 */
struct SimplePoint
{
	float x;
	float y;
};

/**
 * @brief Custom point structure.
 *
 * @param x - x coordinate.
 * @param y - y coordinate.
 * @param sx - x dispersion.
 * @param sy - y dispersion.
 *
 * `sx` and `sy` can be treated as point "size".
 */
struct CustomPoint
{
	float x;
	float y;
	float sx;
	float sy;
};

std::ostream &operator<<(std::ostream &os, const CustomPoint &p);
bool operator<(const CustomPoint &p1, const CustomPoint &p2) noexcept;
bool operator==(const CustomPoint &p1, const CustomPoint &p2) noexcept;

/**
 * @brief A simple point but with direction.
 * @param x - x coordinate.
 * @param y - y coordinate.
 * @param dir - direction (up = 0, right = 1, down = 2, left = 3).
 */
struct DirPoint
{
	float x;
	float y;
	int dir;
};

#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/image.hpp>

#include <vector>
#include <numeric>

static constexpr int maxIntensity = 255;

/**
 * @brief Computes sqrt from vector variance
 *
 * @param mean precomputed vector's mean value
 * @param vec vector to process
 * @return sqrt from vec variance (aka standart deviation, std)
 */
float sqrtVariance(float mean, const std::vector<int> &vec);

/**
 * @brief Calculate the image histogram and write it to `hist`
 *
 * @tparam GrayView - gray image view.
 * @tparam Histogram - random-access container.
 * @param img - source image view.
 * @param hist - histogram array.
 */
template <typename GrayView, typename Histogram>
void customHist(const GrayView &img, Histogram &hist)
{
	for (auto px : img)
		++hist[px];
}

/**
 * @brief Perform Otsu's algorithm to compute an optimal binarization threshold.
 *
 * @see https://www.ipol.im/pub/art/2016/158/
 *
 * @tparam GrayView - gray image view.
 * @param img - source image view.
 * @return Computed threshold.
 */
template <typename GrayView>
int customOtsuThresh(GrayView &img)
{
	// Compute number of pixels
	const int N = img.size();
	int threshold = 0;

	// Compute image histogram
	std::vector<int> hist(256, 0);
	customHist(img, hist);

	// Compute threshold
	// Init variables
	float sum = 0;
	float sumB = 0;
	int q1 = 0;
	int q2 = 0;
	float varMax = 0;
	// Auxiliary value for computing m2
	for (int i = 0; i <= maxIntensity; ++i)
		sum += i * hist[i];
	for (int i = 0; i <= maxIntensity; ++i)
	{
		// Update q1
		q1 += hist[i];
		if (q1 == 0)
			continue;
		// Update q2
		q2 = N - q1;
		if (q2 == 0)
			break;
		// Update m1 and m2
		sumB += i * hist[i];
		const float m1 = sumB / q1;
		const float m2 = (sum - sumB) / q2;
		// Update the between class variance
		float varBetween = static_cast<float>(q1) * static_cast<float>(q2) * (m1 - m2) * (m1 - m2);
		// Update the threshold if necessary
		if (varBetween > varMax)
		{
			varMax = varBetween;
			threshold = i;
		}
	}
	return threshold;
}

/**
 * @brief Binarize image depending on the given threshold.
 *
 * @tparam GrayView - gray image view.
 * @param img - source image view.
 * @param threshold - threshold value.
 */
template <typename GrayView>
void customBinarize(GrayView &img, int threshold)
{
	for(auto &px: img)
		(px > threshold) ? px = maxIntensity : px = 0;
}

/**
 * @brief Apply dilate filter on source image and write result to destination.
 *
 * @tparam GrayView - gray image view.
 * @param src - source image view.
 * @param dst - destination view.
 * @param kernelSize - size of the window over which the maximum value is found.
 */
template <typename GrayView>
void customDilate(const GrayView &src, GrayView &dst, int kernelSize = 2)
{
	using xy_locator = typename GrayView::xy_locator;
	using cached_location_t = typename xy_locator::cached_location_t;
	using boost::gil::gray8_pixel_t; // TODO: point type should be determined by GrayView type

	xy_locator loc = src.xy_at(0, 0);

	std::vector<cached_location_t> locs(kernelSize * kernelSize);
	for (int i = 0; i < locs.size(); ++i)
		locs[i] = loc.cache_location(-i % kernelSize, -i / kernelSize);

	std::vector<gray8_pixel_t> values(kernelSize * kernelSize);

	xy_locator dstLoc = dst.xy_at(0, 0);
	dstLoc(0, 0) = loc(0, 0);

	for (int i = 0; i < src.height(); ++i)
	{
		for (int j = 0; j < src.width(); ++j)
		{
			for (int ii = 0; ii < kernelSize * kernelSize; ++ii)
				values[ii] = loc[locs[ii]];
			if (i == 0)
				dstLoc(0, 0) = *std::max_element(std::begin(values), std::end(values) - 2);
			else
				dstLoc(0, 0) = *std::max_element(std::begin(values), std::end(values));
			++loc.x();
			++dstLoc.x();
		}
		++loc.y();
		++dstLoc.y();
		loc.x() -= src.width();
		dstLoc.x() -= src.width();
	}
}

/**
 * @brief Search starting points for the Theo Pavlidis' algorithm.
 *
 * @tparam GrayView - gray image view.
 * @param src - source image view.
 * @param dst - resulting vector of starting points.
 */
template <typename GrayView>
void getStarts(GrayView &src, std::vector<DirPoint> &dst)
{
	using xy_locator = typename GrayView::xy_locator;
	using cached_location_t = typename xy_locator::cached_location_t;

	xy_locator loc = src.xy_at(1, 1);

	std::vector<cached_location_t> locs(9);
	for (int i = 0; i < 9; ++i)
		locs[i] = loc.cache_location(i / 3 - 1, i % 3 - 1);

	// 0 3 6
	// 1 * 7
	// 2 5 8
	constexpr int all[4][3] = {
		{1, 2, 8}, // Up
		{0, 2, 3}, // Right
		{0, 6, 7}, // Down
		{5, 6, 8}  // Left
	};

	DirPoint extPoint;

	for (int i = 1; i < src.height() - 1; ++i)
	{
		for (int j = 1; j < src.width() - 1; ++j)
		{
			for (int dir = 0; dir < 4; ++dir)
			{
				bool found = true;
				const int *arr = all[dir];
				for (int m = 0; m < 3; ++m)
					found *= (int(loc[locs[arr[m]]]) == 0);
				if (found && (int(loc(0, 0)) == maxIntensity))
				{
					extPoint.x = j;
					extPoint.y = i;
					extPoint.dir = dir;
					dst.push_back(extPoint);
				}
			}
			++loc.x();
		}
		++loc.y();
		loc.x() -= (src.width() - 2);
	}
}

/**
 * @brief Get contour center points from the image using Theo Pavlidis
 * algorithm.
 *
 * @tparam GrayView - gray image view.
 * @param src - source image view.
 * @param starts - starting points obtained with `getStarts` method.
 * @param dst - resulting vector of contour centers and dispersions.
 */
template <typename GrayView>
void getPoints(GrayView &src, std::vector<DirPoint> &starts, std::vector<CustomPoint> &dst)
{
	using xy_locator = typename GrayView::xy_locator;
	using cached_location_t = typename xy_locator::cached_location_t;

	xy_locator loc = src.xy_at(1, 1);
	std::vector<cached_location_t> locs(9);

	std::vector<SimplePoint> movements(9);
	for (int i = 0; i < 9; ++i)
	{
		locs[i] = loc.cache_location(i / 3 - 1, i % 3 - 1);
		movements[i].x = i / 3 - 1;
		movements[i].y = i % 3 - 1;
	}
	// -1 	0 	1
	//
	// 0 	3 	6	// -1
	// 1 	* 	7	// 0
	// 2 	5 	8	// 1

	// up, right, down, left
	// 0   1      2     3

	// right turns:
	// up -> right
	// right -> down
	// down -> left
	// left -> up

	constexpr int all[4][3] = {
		{0, 3, 6}, // Up
		{6, 7, 8}, // Right
		{8, 5, 2}, // Down
		{2, 1, 0}  // Left
	};

	for (const DirPoint &extPoint : starts)
	{
		int curDir = extPoint.dir;
		int currentX = extPoint.x;
		int currentY = extPoint.y;

		loc.x() += currentX - 1;
		loc.y() += currentY - 1;

		std::vector<int> coordsX;
		std::vector<int> coordsY;
		while (true)
		{
			bool foundPixel = false;
			for (int i = 0; i < 3; ++i)
			{
				if (loc[locs[all[curDir][i]]] == maxIntensity)
				{
					currentX += movements[all[curDir][i]].x;
					currentY += movements[all[curDir][i]].y;
					loc.x() += int(movements[all[curDir][i]].x);
					loc.y() += int(movements[all[curDir][i]].y);
					foundPixel = true;
					if (i == 0)
						curDir = (curDir + 3) % 4; // turn left
					break;
				}
			}
			if (!foundPixel)
			{
				curDir = (curDir + 5) % 4; // turn right
				continue;
			}
			coordsX.push_back(currentX);
			coordsY.push_back(currentY);
			// TODO: replace mean and variance calculation by Boost accumulators or online Welford algorithm
			if ((currentX == extPoint.x) && (currentY == extPoint.y))
			{
				CustomPoint customPoint;
				customPoint.x = std::accumulate(coordsX.begin(), coordsX.end(), 0.0) / coordsX.size();
				customPoint.y = std::accumulate(coordsY.begin(), coordsY.end(), 0.0) / coordsY.size();
				customPoint.sx = sqrtVariance(customPoint.x, coordsX);
				customPoint.sy = sqrtVariance(customPoint.y, coordsY);
				dst.push_back(customPoint);
				loc.x() -= currentX - 1;
				loc.y() -= currentY - 1;
				break;
			}
		}
	}
}

#endif // CUSTOM_CV_H
