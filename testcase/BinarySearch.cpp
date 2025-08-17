#include <vector>
#include <algorithm>
#include <numeric>
using namespace::std;

class BinarySearchSolution {
public:
  // https://leetcode.cn/problems/binary-search/
  // 标准库解法
  int search_std(vector<int>& nums, int target) {
    // Searches for the first element in the partitioned range [first, last) which is not ordered before value.
    // Returns the first iterator iter in [first, last) where bool(*iter < value) is false, or last if no such iter exists.
    auto it = std::lower_bound(nums.cbegin(), nums.cend(), target);
    if (it == nums.cend() || *it != target) {
      return -1;
    }
    return it - nums.cbegin();
  }

  int search_tradition(vector<int>& nums, int target) {
    int i = 0, j = nums.size(); // [l, r)
    while (i < j) {
      int mid = i + ((j-i)>>1);
      if (nums[mid] == target) {
        return mid;
      } else if (nums[mid] < target) {
        i = mid + 1;
      } else {
        j = mid;
      }
    }
    return -1;
  }
  int search_from_lowerBound(vector<int>& nums, int target) {
    int i = 0, j = nums.size(); // [l, r)
    auto f = [target](int elementValue) {return elementValue < target;};

    // f(i) true, f(i+1) false, f(i+...) false
    while (i < j) {
      int mid = i + ((j-i)>>1);
      if (f(nums[mid])) { // element < target is true
        i = mid + 1;
      } else { // element < target is false
        j = mid;
      }
    }
    if (j < 0 || j >= nums.size() || nums[j] != target) {
      return -1;
    }
    // i == j == mid
    return j;
  }

  // https://leetcode.cn/problems/search-insert-position/
  // 给定一个排序数组 nums 和一个目标值 target，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
  // 请必须使用时间复杂度为 O(log n) 的算法。
  int searchInsert(vector<int>& nums, int target) {
    int i = 0, j = nums.size();
    auto f = [target](int ele) {return ele < target;};
    while (i < j) {
      int mid = i + ((j - i) >> 1);
      if (f(nums[mid])) {
        i = mid + 1;
      } else {
        j = mid;
      }
    }
    return j;

    // 分析如下:
    // for any non empty array, j must in [0, N]
    // j < 0 || j >= nums.size() || nums[j] != target
    /*
    if (j >= nums.size() || nums[j] != target) { // not found
      return j;
    }
    return j; // found
    */
  }

  // https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
  // 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
  // 34. 在排序数组中查找元素的第一个和最后一个位置
  // 如果数组中不存在目标值 target，返回 [-1, -1]。
  vector<int> searchRange(vector<int>& nums, int target) {
    int i = 0, j = nums.size();

    // 找到第一个元素 >= target, ele < target 为false
    // true, ..., true, false
    //            0   , 3, 3, 3, 5 target=3
    auto f = [target](int ele) {return ele < target;};
    while (i < j) {
      int mid = i + ((j - i) >> 1);
      if (f(nums[mid])) {
        i = mid + 1;
      } else {
        j = mid;
      }
    }
    if (j >= nums.size() || nums[j] != target) {
      return {-1, -1};
    }
    // [j, ?]

    // 找到第一个元素 大于 target, target < element 为true
    // false, ..., false, true
    // 0 3     3      3, 5 target = 3
    auto gUpper = [target](int ele) {return ele > target;};
    int l = j, r = nums.size();
    while (l < r) {
      int mid = l + ((r - l) >> 1);
      if (gUpper(nums[mid])) {
        r = mid;
      } else {
        l = mid + 1;
      }
    }
    return {j, r - 1};
  }

  // https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/
  // 134 寻找旋转排序数组中的最小值, 题意确保 nums 无重复元素
  int findMin(vector<int>& nums) {
    const int N  = nums.size();
    if (N == 0) {
      return -1; // 特殊处理
    }
    int i = 0, j = N - 1;
    while (i < j) {
      int pivot = i + ((j - i)>>1);
      if (nums[j] < nums[pivot]) {
        i = pivot + 1;
      } else {
        j = pivot;
      }
    }
    return nums[i];
  }

  // https://leetcode.cn/problems/search-in-rotated-sorted-array/description/
  // 搜索旋转排序数组
  /*整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 向左旋转 3 次后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
   */
  int search_in_rotated_ascending_array(vector<int>& nums, int target) {
    const int N = nums.size();
    if (N == 0) {
      return -1;
    }
    int rotatedIndex = 0; // 先找旋转点
    int minV = nums[0];
    int i = 0, j = N;
    while (i < j) {
      int mid = i + ((j - i) >> 1);
      if (nums[i] < nums[mid]) {
        if (nums[i] < minV) { // a[]={2,3,0,1}
          minV = nums[i];
          rotatedIndex = i;
        }
        i = mid + 1;
      } else {
        if (nums[mid] < minV) {
          minV = nums[mid];
          rotatedIndex = mid;
        }
        j = mid;
      }
    }
    // [0, rotatedIndex) +
    // [rotatedIndex, N) +
    if (rotatedIndex != 0 && target >= nums[0] && target <= nums[rotatedIndex - 1]) {
      auto it = std::lower_bound(nums.cbegin(), nums.cbegin()+rotatedIndex, target);
      if (it == nums.cend() || *it != target) {
        return -1;
      }
      return it - nums.cbegin();
    } else if (target >= nums[rotatedIndex] && target <= nums.back()) {
      auto it = std::lower_bound(nums.cbegin()+rotatedIndex, nums.cend(), target);
      if (it == nums.cend() || *it != target) {
        return -1;
      }
      return it - nums.cbegin();
    }
    return -1;
  }


  // https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/description/
  // 154. 寻找旋转排序数组中的最小值 II
  // 给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
  // 你必须尽可能减少整个过程的操作步骤。
  int findMin2(vector<int>& nums) {
    const int N = nums.size();
    if (N == 0) {
      return INT_MIN; // 特殊处理, 暂时返回-1
    }
    int i = 0, j = N - 1;
    while (i < j) {
      int pivot = i + ((j-i)>>1);
      if (nums[j] == nums[pivot]) {
        --j;
      } else if (nums[j] < nums[pivot]) {
        i = pivot + 1;
      } else {
        j = pivot;
      }
    }
    return nums[i];
  }

  // https://leetcode.cn/problems/search-a-2d-matrix-ii/
  // 240. 搜索二维矩阵 II
  /*编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。*/
  bool searchMatrix(vector<vector<int>>& matrix, int target) {
    const int M = matrix.size();
    if (M == 0) {
      return false;
    }
    const int N = matrix.front().size();
    if (N == 0) {
      return false;
    }
    int i = 0;
    int j = N - 1;

    while (i < M && j > -1) {
      if (matrix[i][j] == target) {
        return true;
      } else if (target < matrix[i][j]) {
        --j;
      } else {
        ++i;
      }
    }
    return false;
  }


  // https://leetcode.cn/problems/median-of-two-sorted-arrays
  // 4. 寻找两个正序数组的中位数
  // 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
  double f(const vector<int>& nums1, const vector<int>& nums2, int k) {
    // 要获取c[k], 前面要有k个元素, 取a[k/2-1], b[k/2-1], 含端点共有(k/2) * 2 <= k个元素
    if (nums1.empty()) {
      return nums2[k];
    }
    if (nums2.empty()) {
      return nums1[k];
    }
    if (k == 0) {
      return std::min(nums1.front(), nums2.front());
    } else if (k == 1) {
      if (nums1.front() == nums2.front()) {
        return nums1.front();
      } else if (nums1.front() < nums2.front()) {
        return nums1.size() > 1 ? std::min(nums1[1], nums2.front()) : nums2.front();
      } else {
        return nums2.size() > 1 ? std::min(nums2[1], nums1.front()) : nums1.front();
      }
    }
    int index = k / 2 - 1;
    int index1 = index, index2 = index;
    // index1 + 1 + index2 + 1 = k
    if (nums1.size() <= index) {
      index1 = nums1.size() - 1;
      index2 = k - 2 - index1;
    } else if (nums2.size() <= index) {
      index2 = nums2.size() - 1;
      index1 = k - 2 - index2;
    }
    if (nums1[index1] <= nums2[index2]) {
      // 排除nums[0, index1]部分
      vector<int> nums1Temp{nums1.cbegin() + index1 + 1, nums1.cend()};
      return f(nums1Temp, nums2, k - index1 - 1);
    } else {
      vector<int> nums2Temp{nums2.cbegin() + index2 + 1, nums2.cend()};
      return f(nums1, nums2Temp, k - index2 - 1);
    }
  }

  double fIteration(const vector<int>& nums1, const vector<int>& nums2, int k) {
    int aDev = 0, bDev = 0;
    const int TotalK = k;
    while (true) {
      if (aDev == nums1.size()) {
        return nums2[bDev + k];
      }
      if (bDev == nums2.size()) {
        return nums1[aDev + k];
      }
      if (k == 0) {
        return std::min(nums1[aDev], nums2[bDev]);
      }
      /* 可简化如下分支
      else if (k == 1) {
        if (nums1[aDev] == nums2[bDev]) {
          return nums1[aDev];
        } else if (nums1[aDev] < nums2[bDev]) {
          return nums1.size() > aDev + 1 ? std::min(nums1[aDev + 1], nums2[bDev]) : nums2[bDev];
        } else {
          return nums2.size() > bDev + 1 ? std::min(nums2[bDev + 1], nums1[aDev]) : nums1[aDev];
        }
      }
      */
      int index = k == 1 ? 0 : k / 2 - 1;
      // 简化版本, 一个取到末尾, 另一个任然是k/2-1
      int index1 = std::min((int)nums1.size() - 1, aDev + index);
      int index2 = std::min((int)nums2.size() - 1, bDev + index);
      // 完整版本, 一个取到末尾, 另一个自适应取到 TotalK - (size - 1)
      // int index1 = aDev + index, index2 = bDev + index; // 全局索引
      // index1 + 1 + index2 + 1 = k
      // if (nums1.size() <= index1) {
      //   index1 = nums1.size() - 1;
      //   index2 = TotalK - 2 - index1; // TotalK - (num1.size() + 1)
      // } else if (nums2.size() <= index2) {
      //   index2 = nums2.size() - 1;
      //   index1 = TotalK - 2 - index2;
      // }
      if (nums1[index1] <= nums2[index2]) {
        // 排除了nums1[0, index1]部分
        // 排除nums1[aDev, index1]部分
        k -= (index1 - aDev + 1);
        aDev = index1 + 1;
      } else {
        k -= (index2 - bDev + 1);
        bDev = index2 + 1;
      }
    }
  }

  double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    const int M = nums1.size(), N = nums2.size();
    const int count = M + N;
    // if (count & 1) { // odd
    //   return f(nums1, nums2, count / 2);
    // } else { // even
    //   return (f(nums1, nums2, count / 2 - 1) + f(nums1, nums2, count / 2)) / 2.0;
    // }
    if (count & 1) { // odd
      return fIteration(nums1, nums2, count / 2);
    } else { // even
      return (fIteration(nums1, nums2, count / 2 - 1) + fIteration(nums1, nums2, count / 2)) / 2.0;
    }
  }
};


#include "gtest/gtest.h"
class BinarySearchTest : public testing::Test {
  void SetUp() override {
  }
};

TEST_F(BinarySearchTest, std_ok){
  vector<int> nums{0,1,2,3,4,5,6,7,8,9};
  BinarySearchSolution bss;
  EXPECT_EQ(bss.search_std(nums, 0), 0);
  EXPECT_EQ(bss.search_std(nums, 1), 1);
  EXPECT_EQ(bss.search_std(nums, 5), 5);
  EXPECT_EQ(bss.search_std(nums, 9), 9);

  EXPECT_EQ(bss.search_std(nums, -1), -1);
  EXPECT_EQ(bss.search_std(nums, 10), -1);
}

TEST_F(BinarySearchTest, tradition_ok){
  vector<int> nums{0,1,2,3,4,5,6,7,8,9};
  BinarySearchSolution bss;
  EXPECT_EQ(bss.search_tradition(nums, 0), 0);
  EXPECT_EQ(bss.search_tradition(nums, 1), 1);
  EXPECT_EQ(bss.search_tradition(nums, 5), 5);
  EXPECT_EQ(bss.search_tradition(nums, 9), 9);

  EXPECT_EQ(bss.search_tradition(nums, -1), -1);
  EXPECT_EQ(bss.search_tradition(nums, 10), -1);
}

TEST_F(BinarySearchTest, from_lowerBound_ok){
  vector<int> nums{0,1,2,3,4,5,6,7,8,9};
  BinarySearchSolution bss;

  EXPECT_EQ(bss.search_from_lowerBound(nums, 0), 0);
  EXPECT_EQ(bss.search_from_lowerBound(nums, 1), 1);
  EXPECT_EQ(bss.search_from_lowerBound(nums, 5), 5);
  EXPECT_EQ(bss.search_from_lowerBound(nums, 9), 9);

  EXPECT_EQ(bss.search_from_lowerBound(nums, -1), -1);
  EXPECT_EQ(bss.search_from_lowerBound(nums, 10), -1);
}

TEST_F(BinarySearchTest, index_or_insert_index_ok){
  vector<int> nums{0,1,2,3,4,5,6,7,8,9};
  BinarySearchSolution bss;
  EXPECT_EQ(bss.searchInsert(nums, 0), 0);
  EXPECT_EQ(bss.searchInsert(nums, 1), 1);
  EXPECT_EQ(bss.searchInsert(nums, 5), 5);
  EXPECT_EQ(bss.searchInsert(nums, 9), 9);

  EXPECT_EQ(bss.searchInsert(nums, -1), 0);
  EXPECT_EQ(bss.searchInsert(nums, 10), 10);
}

TEST_F(BinarySearchTest, searchRange_ok){
  BinarySearchSolution bss;
  vector<int> nums{0,1,2,3,4,5,6,7,8,9};
  EXPECT_EQ(bss.searchRange(nums, 0), (vector<int>{0, 0}));
  EXPECT_EQ(bss.searchRange(nums, 1), (vector<int>{1, 1}));
  EXPECT_EQ(bss.searchRange(nums, 5), (vector<int>{5, 5}));
  EXPECT_EQ(bss.searchRange(nums, 9), (vector<int>{9, 9}));

  EXPECT_EQ(bss.searchRange(nums, -1), (vector<int>{-1, -1}));
  EXPECT_EQ(bss.searchRange(nums, 10), (vector<int>{-1, -1}));


  vector<int> numsWithDup1{1, 1};
  EXPECT_EQ(bss.searchRange(numsWithDup1, 1), (vector<int>{0, 1}));

  vector<int> numsWithDup{0,1,1,1,2};
  EXPECT_EQ(bss.searchRange(numsWithDup, 1), (vector<int>{1, 3}));
}

TEST_F(BinarySearchTest, findMin_ok){
  BinarySearchSolution bss;
  vector<int> nums0{0,1,2,3,4,5,6,7};
  vector<int> nums1{7,0,1,2,3,4,5,6};
  vector<int> nums2{6,7,0,1,2,3,4,5};
  vector<int> nums3{5,6,7,0,1,2,3,4};
  vector<int> nums4{4,5,6,7,0,1,2,3};
  vector<int> nums5{3,4,5,6,7,0,1,2};
  vector<int> nums6{2,3,4,5,6,7,0,1};
  vector<int> nums7{1,2,3,4,5,6,7,0};
  EXPECT_EQ(bss.findMin(nums0), 0);
  EXPECT_EQ(bss.findMin(nums1), 0);
  EXPECT_EQ(bss.findMin(nums2), 0);
  EXPECT_EQ(bss.findMin(nums3), 0);
  EXPECT_EQ(bss.findMin(nums4), 0);
  EXPECT_EQ(bss.findMin(nums5), 0);
  EXPECT_EQ(bss.findMin(nums6), 0);
  EXPECT_EQ(bss.findMin(nums7), 0);
}

TEST_F(BinarySearchTest, findMin2_ok) {
  BinarySearchSolution bss;
  vector<int> nums0{0,0,1,2};
  vector<int> nums1{2,0,0,1};
  vector<int> nums2{1,2,0,0};
  EXPECT_EQ(bss.findMin2(nums0), 0);
  EXPECT_EQ(bss.findMin2(nums1), 0);
  EXPECT_EQ(bss.findMin2(nums2), 0);
}

TEST_F(BinarySearchTest, search_in_rotated_ascending_array_ok){
  BinarySearchSolution bss;
  vector<int> nums0{0,1,2,3,4,5,6,7};
  vector<int> nums1{7,0,1,2,3,4,5,6};
  vector<int> nums2{6,7,0,1,2,3,4,5};
  vector<int> nums3{5,6,7,0,1,2,3,4};
  vector<int> nums4{4,5,6,7,0,1,2,3};
  vector<int> nums5{3,4,5,6,7,0,1,2};
  vector<int> nums6{2,3,4,5,6,7,0,1};
  vector<int> nums7{1,2,3,4,5,6,7,0};
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums0, 0), 0);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums1, 0), 1);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums2, 0), 2);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums3, 0), 3);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums4, 0), 4);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums5, 0), 5);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums6, 0), 6);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums7, 0), 7);

  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums0, -1), -1);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums0, 10), -1);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums7, -1), -1);
  EXPECT_EQ(bss.search_in_rotated_ascending_array(nums7, 10), -1);
}

TEST_F(BinarySearchTest, searchMatrix_ok) {
  BinarySearchSolution bss;
  vector<vector<int>> matrix{{0,1,2},
                             {3,4,5},
                             {6,7,8}};
  EXPECT_EQ(bss.searchMatrix(matrix, -1), false);
  EXPECT_EQ(bss.searchMatrix(matrix, 10), false);

  EXPECT_EQ(bss.searchMatrix(matrix, 0), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 1), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 2), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 3), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 4), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 5), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 6), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 7), true);
  EXPECT_EQ(bss.searchMatrix(matrix, 8), true);

  vector<vector<int>> matrix2 = {{1,4,7,11,15},
                                 {2,5,8,12,19},
                                 {3,6,9,16,22},
                                 {10,13,14,17,24},
                                 {18,21,23,26,30}};
  EXPECT_EQ(bss.searchMatrix(matrix2, 5), true);
}


TEST_F(BinarySearchTest, findMedianSortedArrays_ok) {
  BinarySearchSolution bss;

  vector<int> aEmpty, aNonEmpty{1, 2};
  vector<int> bEmpty, bNonEmpty{1, 2};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(aEmpty, bNonEmpty) - 1.5), 1E-6);
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(aNonEmpty, bEmpty) - 1.5), 1E-6);

  vector<int> a{0,1,2};
  vector<int> b{3,4};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(a, b) - 2), 1E-6);
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(b, a) - 2), 1E-6);

  vector<int> aEven{0,1,2};
  vector<int> bEven{3,4,6};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(aEven, bEven) - 2.5), 1E-6);
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(bEven, aEven) - 2.5), 1E-6);

  vector<int> a7 {2,3,4,5,6,7,8};
  vector<int> b1 {1};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(a7, b1) - 4.5), 1E-6);
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(b1, a7) - 4.5), 1E-6);

  vector<int> a4 {1,3,5,6};
  vector<int> b4 {2,4,6,8};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(a4, b4) - 4.5), 1E-6);
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(b4, a4) - 4.5), 1E-6);

  vector<int> a50 {0,0,0,0,0};
  vector<int> b6 {-1,0,0,0,0,0,1};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(a50, b6) - 0), 1E-6);

  vector<int> a15 {1,2,3,4,5};
  vector<int> b617 {6,7,8,9,10,11,12,13,14,15,16,17};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(a15, b617) - 9), 1E-6);
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(b617, a15) - 9), 1E-6);

  vector<int> a9{0,1,2,3,4,5,6,7,8};
  vector<int> b9{-8,-7,-6,-5,-4,-3,-2,-1,0};
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(a9, b9) - 0), 1E-6);
  EXPECT_LE(std::abs(bss.findMedianSortedArrays(b9, a9) - 0), 1E-6);
}
