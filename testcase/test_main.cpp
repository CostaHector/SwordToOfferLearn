#include "Others.h"
#include "gtest/gtest.h"
#include <vector>
using std::vector;
#include <queue>
using std::priority_queue;

class MainTest : public testing::Test {

};

TEST_F(MainTest, f42_ok) {
  EXPECT_EQ(f42(), 42);
}

TEST_F(MainTest, nok) {
  EXPECT_TRUE(3 > 2);
}

// https://leetcode.cn/problems/container-with-most-water/description/
class SolutionmaxArea {
 public:
  int maxArea(vector<int>& height) {
    int N = (int) height.size();
    if (N < 2) {
      return 0;
    }
    // violence double pointer
    int l = 0, r = N - 1;
    int maxAreaVal = 0;
    while (l < r) {
      maxAreaVal = std::max(maxAreaVal, (r - l) * std::min(height[r], height[l]));
      // width fixed: currentWidth - 1
      // next maxArea should be in max height one, so we ++ or -- the lowest index now to let column to be as high as possible
      if (height[l] < height[r]) {
        ++l;
      } else {
        --r;
      }
    }
    return maxAreaVal;
  }
  // https://leetcode.cn/problems/trapping-rain-water/
  int trapPriorityQueue(vector<int>& height) { // priorityQueue
    int N = (int) height.size();
    if (N < 3) {
      return 0;
    }
    // currentWater = min(leftMaxHeight, rightMaxHeight) - currentHeight
    // ans = sum{currentWater_i|1<=i<=N-2}
    vector<int> leftMaxHeight, rightMaxHeight; // f[i] max height in (...,i), g[i] max height in (i, ...)
    leftMaxHeight.reserve(N + 1);
    rightMaxHeight.reserve(N + 1);
    leftMaxHeight.push_back(0);
    rightMaxHeight.push_back(0);
    priority_queue<int> pq, rpq;
    for (int currentHeight: height) {
      pq.push(currentHeight);
      leftMaxHeight.push_back(pq.top());
    }
    for (auto it = height.crbegin(); it < height.crend(); ++it) {
      rpq.push(*it);
      rightMaxHeight.push_back(rpq.top());
    }
    int ans{0};
    for (int i = 1; i < N - 1; ++i) {
      int leftRightMin = std::min(leftMaxHeight[i], rightMaxHeight[N - 1 - i]);
      if (leftRightMin > height[i]) {
        ans += leftRightMin - height[i];
      }
    }
    return ans;
  }
  int trap(vector<int>& height) { // priorityQueue
    int N = (int) height.size();
    if (N < 3) {
      return 0;
    }
    // currentWater = min(leftMaxHeight, rightMaxHeight) - currentHeight
    // ans = sum{currentWater_i|1<=i<=N-2}
    int leftMaxHeight{height[0]}, rightMaxHeight{height[N-1]};
    int ans{0};
    int l = 1, r = N - 2;
    while (l <= r) {
      leftMaxHeight = std::max(leftMaxHeight, height[l]);
      rightMaxHeight = std::max(rightMaxHeight, height[r]);
      if (leftMaxHeight > rightMaxHeight) {
        ans += rightMaxHeight - height[r--];
      }else {
        ans += leftMaxHeight - height[l++];
      }
    }
    return ans;
  }
};

TEST_F(MainTest, maxArea) {
  vector<int> height1{1,8,6,2,5,4,8,3,7};
  EXPECT_EQ(SolutionmaxArea().maxArea(height1), 7 * 7); // index at 1 and last one. height = min(8, 7)=7
  vector<int> height2{6,8};
  EXPECT_EQ(SolutionmaxArea().maxArea(height2), 1 * 6);
  vector<int> height3{8,8};
  EXPECT_EQ(SolutionmaxArea().maxArea(height3), 1 * 8);
  vector<int> height4{8,1,8};
  EXPECT_EQ(SolutionmaxArea().maxArea(height4), 2 * 8);
}

TEST_F(MainTest, trap) {
  vector<int> height1{5,0,4,2,3,5,1};
  EXPECT_EQ(SolutionmaxArea().trap(height1), 5+1+3+2);
  vector<int> height2{0,1,0,2,1,0,1,3,2,1,2,1};
  EXPECT_EQ(SolutionmaxArea().trap(height2), 6);
  vector<int> height3{2,0,2};
  EXPECT_EQ(SolutionmaxArea().trap(height3), 2);
}


