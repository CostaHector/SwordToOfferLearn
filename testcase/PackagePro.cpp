#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include "gtest/gtest.h"

using namespace::std;

class PackageProSolution {
public:
  /*
设dp[i][j]表示容量为i的背包, 可选[0,j)索引内的物品时, 背包的最大重量,
定义vector<vector<int>>dp(N+1, vector<int>(nums.size()+1, 0));
那么转移方程就是 (外层i, 里层j)
dp[i][j] = max{dp[i][j-1], i-nums[j-1] >=0 ? dp[i-nums[j-1]][j-1] +nums[j-1] : 0}; i>=1 && j >=1;
dp[i][j] = 0; i==0||j==0;
因此如下的正序遍历就行了
*/
  int maxWeight(const vector<int>& nums, int N) {
    // vector<vector<int>> dp(N + 1, vector<int>(nums.size() + 1, 0));
    // for (int i = 1; i < N + 1; ++i){
    //   for (int j = 1; j < nums.size() + 1; ++j){
    //     if (i >= nums[j-1]) {
    //       dp[i][j] = max(dp[i][j-1], dp[i - nums[j-1]][j-1] + nums[j-1]);
    //     } else {
    //       dp[i][j] = dp[i][j-1];
    //     }
    //   }
    // }
    // return dp.back().back();
    // 注意到dp只取决于左侧, 以及左上, 这里可以去除j维度;
    // 同时要考虑到, 没了j维度区分上一可选索引后, dp[i] i的遍历顺序要改为从大到小,
    // 否则 先更新了dp'[0], dp'[1]想用左上角的dp[0], 实际上用到是是dp'[0] 造成完全背包
    vector<int> dp1(N+1, 0);
    // for (int j = 1; j < nums.size() + 1; ++j) {
    for (auto X: nums) {
      // for (int i = nums[j - 1]; i < N + 1; ++i) { // 这是完全背包, 因为此时dp[b]可能直接来自正上方的dp[a], 多次使用了nums[j-1]
      for (int i = N; i >= X; --i) { // 这是0-1背包, 先算大索引
        dp1[i] = std::max(dp1[i], dp1[i-X] + X);
      }
    }
    return dp1.back();
  }


  bool canPartition(const vector<int>& nums) {
    int s = std::accumulate(nums.cbegin(), nums.cend(), 0);
    if (s & 1) {
      return false;
    }
    if (s == 0) {
      return true;
    }
    int N = s / 2;
    vector<int> dp1(N+1, 0);
    for (int j = 1; j < nums.size() + 1; ++j) {
      for (int i = N ; i >= nums[j - 1]; --i) {
        dp1[i] = std::max(dp1[i], dp1[i-nums[j-1]] + nums[j-1]);
        if (dp1[i] == N) {
          return true;
        }
      }
    }
    return false;
  }
  // https://leetcode.cn/problems/last-stone-weight/
  /*
有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。
*/
  int lastStoneWeight(const vector<int>& stones) {
    std::priority_queue<int, vector<int>, std::less<int>> pri{std::less<int>(), stones};
    while (pri.size() >= 2) {
      int largest = pri.top();
      pri.pop();
      int second2Largest = pri.top();
      pri.pop();
      if (largest != second2Largest) {
        pri.push(largest - second2Largest);
      }
    }
    return pri.empty() ? 0 : pri.top();
  }
  int lastStoneWeightII(const vector<int>& stones) {
    if (stones.empty()) {
      return 0;
    }
    if (stones.size() == 1) {
      return stones.front();
    }
    int s = std::accumulate(stones.cbegin(), stones.cend(), 0);
    int N = s / 2;
    // vector<vector<int>> dp(N+1, vector<int>(stones.size()+1, 0));
    // for (int i = 1; i < N + 1; ++i) {
    //   for (int j = 1; j < stones.size() + 1; ++j) {
    //     dp[i][j] = dp[i][j-1];
    //     if (i >= stones[j-1]) {
    //       dp[i][j] = std::max(dp[i][j-1], dp[i-stones[j-1]][j-1] + stones[j-1]);
    //     }
    //   }
    // }
    // int maxWeight2Half = dp.back().back();
    // 错误解法请见swordToOffer云记
    vector<int> dp(N+1, 0);
    for (int j = 1; j < stones.size() + 1; ++j) {
      for (int i = N; i >= stones[j-1]; --i) {
        dp[i] = std::max(dp[i], dp[i-stones[j-1]] + stones[j-1]);
      }
    }
    int maxWeight2Half = dp.back();
    return s - maxWeight2Half - maxWeight2Half;
  }

  // https://leetcode.cn/problems/target-sum/
  // 递归回溯
  int findTargetCore(const vector<int>& nums, int i, int sum, int target) {
    if (i >= nums.size()) {
      if (sum == target) {
        return 1;
      }
      return 0;
    }
    return findTargetCore(nums, i + 1, sum + nums[i], target)
           + findTargetCore(nums, i + 1, sum - nums[i], target);
  }

  int findTargetSumWaysRecursive(const vector<int>& nums, int target) {
    return findTargetCore(nums, 0, 0, target);
  }

  int findTargetSumWays(const vector<int>& nums, int target) {
    int s = std::accumulate(nums.cbegin(), nums.cend(), 0);
    // (s - neg) - neg = pos - neg = target
    // neg = (s - target)/2
    // because neg >= 0, so s - target is even and >= 0
    if (s < target || (((s - target) & 1) == 1)) {
      return 0;
    }
    // 级数和为target有正有负不好算, 但是数值和为neg, 就能用0-1背包了
    // for example. nums = {1,1,1,1,1}, target=4, sum=5; no way
    // dp[i][j] 表示背包容量为i, 可选[0,j)索引物品时, 数值和等于 neg 的方法数量
    int N = (s - target) / 2;
    // dp: (N+1) - by - (nums.size() + 1) all zero matrix
    // vector<vector<int>> dp(N+1, vector<int>(nums.size() + 1, 0));
    // // dp[i][j] = dp[i][j-1], 无法选择导致不选nums[j-1]
    // // dp[i][j] = dp[i][j-1] + dp[i-nums[j-1]][j-1], 选择了物品nums[j-1]
    // // dp[0][0] = 1, dp[0][>0] >= 1
    // // 注意这里nums[k] >=0, 对于{0,0,1}, target=1, 背包容量neg为0, 可选前2项时, +0-0; -0+0都行
    // dp[0][0] = 1;
    // for (int i = 0; i < 1 + N; ++i) {
    //   for (int j = 1; j < nums.size() + 1; ++j) {
    //     dp[i][j] = dp[i][j-1];
    //     if (i >= nums[j-1]) {
    //       dp[i][j] += dp[i-nums[j-1]][j-1]; // also dp[i][j-1] + dp[i-nums[j-1]][j-1]
    //     }
    //   }
    // }
    // return dp.back().back();

    // 注意到dp[i][j]状态转移来自左侧1个, 和左1上x格
    // 法1: 考虑用两个一维数组简化
    // 分别表示左侧列dp0, 右侧列dp1, 遍历时应该固定j, 让i去递增
    // 法2: 也可以用前面的方法, 倒顺序遍历i
    // vector<int> dp0(1+N, 0), dp1(1+N, 0);
    // dp0[0] = 1;
    // for (int j = 1; j < nums.size() + 1; ++j) {
    //   for (int i = 0; i < 1 + N; ++i) {
    //     dp1[i] = dp0[i];
    //     if (i >= nums[j-1]) {
    //       dp1[i] += dp0[i-nums[j-1]];
    //     }
    //   }
    //   dp0.swap(dp1);
    // }
    // return dp0.back();
    vector<int> dp(1+N, 0);
    dp[0] = 1;
    for (int j = 1; j < nums.size() + 1; ++j) {
      for (int i = N; i >= nums[j-1]; --i) {
        dp[i] += dp[i-nums[j-1]];
      }
    }
    return dp.back();
  }

  // https://leetcode.cn/problems/ones-and-zeroes/description/
  /*给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。*/
  int findMaxForm(const vector<string>& strs, int m, int n) {
    // dp[i][j][k] i,j表示背包容量, i个0, j个1, k表示[0,k)可选索引范围时, 最大子集的长度
    // dp: an (m+1) - by - (n+1) - by - (nums.size()+1) matrix

    // 一个物品的重量(0/1的个数)直接从nums中取
    vector<vector<int>> nums(2, vector<int>(strs.size(), 0));
    for (int i = 0; i < strs.size(); ++i) {
      int zeroCnt = std::count(strs[i].cbegin(), strs[i].cend(), '0');
      nums[0][i] = zeroCnt;
      nums[1][i] = strs[i].size() - zeroCnt;
    }
    /*
 * 转移方程
 * 起始条件
dp[0][0][*] = 0; 背包大小为0, 最大子集长度是空的, 0
dp[i][j][k] = dp[i][j][k-1] 无法选择nums[k-1]
dp[i][j][k] = max{dp[i][j][k-1] 不选择, 1 + dp[i-nums[0][k-1]][j-nums[1][k-1]][k-1] 选择}
*/
    // vector<vector<vector<int>>> dp(m+1, vector<vector<int>>(n+1, vector<int>(strs.size()+1, 0)));
    // for (int i = 0; i < m + 1; ++i) {
    //   for (int j = 0; j < n + 1; ++j) {
    //     for (int k = 1; k < strs.size() + 1; ++k) {
    //       dp[i][j][k] = dp[i][j][k-1];
    //       if (i >= nums[0][k-1] && j >= nums[1][k-1]) {
    //         dp[i][j][k] = std::max(dp[i][j][k], 1 + dp[i-nums[0][k-1]][j-nums[1][k-1]][k-1]);
    //       }
    //     }
    //   }
    // }
    // return dp.back().back().back();

    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int k = 1; k < strs.size() + 1; ++k) {
      for (int i = m; i >= nums[0][k-1]; --i) {
        for (int j = n; j >= nums[1][k-1]; --j) {
          dp[i][j] = std::max(dp[i][j], 1 + dp[i-nums[0][k-1]][j-nums[1][k-1]]);
        }
      }
    }
    return dp.back().back();
  }


  // https://leetcode.cn/problems/coin-change/description/
  /*
322. 零钱兑换

给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。
*/
  int coinChange(const vector<int>& coins, int amount) {
    // // dp[i][j], 背包容量为i(要找零i)时, [0,j)可选索引物品, 所需的最少硬币数
    // vector<vector<int>> dp(amount + 1, vector<int>(coins.size() + 1, INT_MAX));
    // // 无需找零时, 需要的硬币数是0
    // dp[0] = vector<int> (coins.size() + 1, 0);
    // // 其他场景默认最大
    // for (int i = 1; i < amount + 1; ++i) {
    //   for (int j = 1; j < coins.size() + 1; ++j) {
    //     int coinV = coins[j-1];
    //     int minCoins  = dp[i][j-1]; // 不选当前的coinV
    //     for (int k = 1; k * coinV <= i; ++k) {
    //       if (dp[i - k * coinV][j-1] != INT_MAX) {
    //         minCoins  = std::min(minCoins, dp[i - k * coinV][j-1] + k); // 选当前的coinV k次数
    //       }
    //     }
    //     dp[i][j] = minCoins ;
    //   }
    // }
    // return dp.back().back() == INT_MAX ? -1 : dp.back().back();

    // 优化: 观察到dp只取决于左侧, 和左上, 可去除j维度, 再正序遍历i(隐式), 实现完全背包
    vector<int> dp(amount + 1, INT_MAX);
    // 无需找零时, 需要的硬币数是0
    dp[0] = 0;
    // 其他场景默认最大
    for (int j = 1; j < coins.size() + 1; ++j) {
      for (int i = coins[j-1]; i < amount + 1; ++i) { // 正序遍历i(隐式), 实现完全背包
        if (dp[i - coins[j-1]] != INT_MAX){
          dp[i] = std::min(dp[i], dp[i - coins[j-1]] + 1);
        }
      }
    }
    return dp.back() == INT_MAX ? -1 : dp.back();
  }


  int combinationSum4(const vector<int>& coins, int target) {
    // // 顺序无关: 不区分顺序
    // vector<vector<int>> dp(target+1, vector<int>(coins.size()+1, 0));
    // dp[0] = vector<int>(coins.size()+1, 1);
    // for (int i = 1; i < target+1; ++i) {
    //   for (int j = 1; j < coins.size()+1; ++j) {
    //     // 1 方法1
    //     int coin = coins[j-1], waysCnt = 0;
    //     for (int k = 0; k * coin <= i; ++k) {
    //       waysCnt += dp[i - k * coin][j-1];
    //     }
    //     dp[i][j] = waysCnt;

    //     // 2 方法2: 注意此处的dp[i - coin][j这里不要没有-1]
    //     int coin = coins[j - 1];
    //     dp[i][j] = dp[i][j - 1];
    //     if (i >= coin) {
    //       dp[i][j] += dp[i - coin][j];
    //     }
    //   }
    // }
    // return dp.back().back();

    // 顺序无关: 不区分顺序
    // vector<int> dp(target+1, 0);
    // dp[0] = 1;
    // for (int j = 1; j < coins.size()+1; ++j) {
    //   int coin = coins[j-1];
    //   for (int i = coin; i < target+1; ++i) {
    //       dp[i] += dp[i - coin];
    //   }
    // }

    // 顺序相关: 区分顺序
    // 原, 二维dp 先i背包, 后j物品, dp[i,j]=sum(dp[i-k*coin,j-1]) 顺序无关 完全背包,
    //                    也可以是dp[i,j]=dp[i,j-1]+dp[i-coin,j]
    // 改1, 一维dp 先j物品, 后背包i, 背包升序 完全背包 顺序无关 完全背包,
    // 改2, 一维dp 先i背包, 后物品j, 物品升序, 某一行执行时
    vector<int> dp(target+1, 0);
    dp[0] = 1;
    for (int i = 1; i < target+1; ++i) {
      // 这也是跳楼梯题目的解法 f(n) = f(n-1) + f(n-2), f(0) = 1; 一次可以跳1级或者2级;
      // 变态跳台阶
      for (int j = 1; j < coins.size()+1; ++j) {
        if (i >= coins[j-1]) {
          dp[i] += dp[i - coins[j-1]];
        }
      }
    }
    return dp.back();
  }

  // https://leetcode.cn/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/
/*
LCR 127. 跳跃训练

今天的有氧运动训练内容是在一个长条形的平台上跳跃。平台有 num 个小格子，每次可以选择跳 一个格子 或者 两个格子。请返回在训练过程中，学员们共有多少种不同的跳跃方式。

结果可能过大，因此结果需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
*/
  int trainWays(int num) {
    int dp0 = 1, dp1 = 1;
    for (int i = 2; i < num + 1; ++i) {
      int temp = dp1;
      dp1 = (dp0 + dp1) % 1000000007; // f'(n) = f(n-2) + f(n-1)
      dp0 = temp; // f(n-1)
    }
    return dp1;
  }
};

class PackageProTest: public testing::Test{

};

TEST_F(PackageProTest, maxWeight_ok) {
  PackageProSolution sol;
  EXPECT_EQ(sol.maxWeight({1,2,3}, 6), 6);
  EXPECT_EQ(sol.maxWeight({1,2,3}, 7), 6);
  EXPECT_EQ(sol.maxWeight({1,2,3}, 5), 5);
  EXPECT_EQ(sol.maxWeight({3,2,2}, 7), 7);
  EXPECT_EQ(sol.maxWeight({3,2,2}, 6), 5);
  EXPECT_EQ(sol.maxWeight({3,2,2}, 5), 5);
  EXPECT_EQ(sol.maxWeight({3,2,2}, 4), 4);
  EXPECT_EQ(sol.maxWeight({1,2,5}, 8), 8);
  EXPECT_EQ(sol.maxWeight({1,2}, 8), 3); // 防止完全背包

  vector<int> nums1 = {1, 2, 3, 4};
  EXPECT_EQ(sol.maxWeight(nums1, 5), 5);

  vector<int> nums2 = {2, 3, 5, 7};
  EXPECT_EQ(sol.maxWeight(nums2, 10), 10);

  vector<int> nums3 = {1, 1, 1, 1};
  EXPECT_EQ(sol.maxWeight(nums3, 3), 3);

  vector<int> nums4 = {4, 6, 8};
  EXPECT_EQ(sol.maxWeight(nums4, 5), 4);
}

TEST_F(PackageProTest, canPartition_ok) {
  PackageProSolution sol;
  EXPECT_TRUE(sol.canPartition({1,1}));
  EXPECT_TRUE(sol.canPartition({1,2,3}));
  EXPECT_TRUE(sol.canPartition({1,2,2,2,3}));
  EXPECT_FALSE(sol.canPartition({1,2,5}));
  EXPECT_FALSE(sol.canPartition({1,2}));
  EXPECT_FALSE(sol.canPartition({1}));
  EXPECT_FALSE(sol.canPartition({2}));
}

TEST_F(PackageProTest, lastStoneWeight_ok) {
  PackageProSolution sol;
  EXPECT_EQ(sol.lastStoneWeight({3,3}), 0);
  EXPECT_EQ(sol.lastStoneWeight({3}), 3);
  EXPECT_EQ(sol.lastStoneWeight({1,2}), 1);
  EXPECT_EQ(sol.lastStoneWeight({3,1,2}), 0);
  EXPECT_EQ(sol.lastStoneWeight({1,5,3,7}), 0);
  EXPECT_EQ(sol.lastStoneWeight({4,3,4,3,2}), 2);
}

TEST_F(PackageProTest, lastStoneWeight2_ok) {
  PackageProSolution sol;
  EXPECT_EQ(sol.lastStoneWeightII({3,3}), 0);
  EXPECT_EQ(sol.lastStoneWeightII({3}), 3);
  EXPECT_EQ(sol.lastStoneWeightII({1,2}), 1);
  EXPECT_EQ(sol.lastStoneWeightII({3,1,2}), 0);
  EXPECT_EQ(sol.lastStoneWeightII({1,5,3,7}), 0);
  EXPECT_EQ(sol.lastStoneWeightII({4,3,4,3,2}), 0);
  EXPECT_EQ(sol.lastStoneWeightII({31,26,33,21,40}), 5);
  EXPECT_EQ(sol.lastStoneWeightII({11,6,13,1,20}), 1);// 防止完全背包
  EXPECT_EQ(sol.lastStoneWeightII({1,10}), 9); // 防止完全背包, N = 11/2 = 5, S - 2*HALF = 11 - 10 = 1
}

TEST_F(PackageProTest, findTargetSumWays_ok) {
  PackageProSolution sol;
  EXPECT_EQ(sol.findTargetSumWays({1,1,1,1,1}, 6), 0);
  EXPECT_EQ(sol.findTargetSumWays({1,1,1,1,1}, 5), 1);
  EXPECT_EQ(sol.findTargetSumWays({1,1,1,1,1}, 4), 0);
  EXPECT_EQ(sol.findTargetSumWays({1,1,1,1,1}, 3), 5);
  EXPECT_EQ(sol.findTargetSumWays({1,1,1,1,1}, 2), 0);
  EXPECT_EQ(sol.findTargetSumWays({1,1,1,1,1}, 1), 10);
  EXPECT_EQ(sol.findTargetSumWays({1,1,1,1,1}, 0), 0);

  EXPECT_EQ(sol.findTargetSumWaysRecursive({1,1,1,1,1}, 6), 0);
  EXPECT_EQ(sol.findTargetSumWaysRecursive({1,1,1,1,1}, 5), 1);
  EXPECT_EQ(sol.findTargetSumWaysRecursive({1,1,1,1,1}, 4), 0);
  EXPECT_EQ(sol.findTargetSumWaysRecursive({1,1,1,1,1}, 3), 5);
  EXPECT_EQ(sol.findTargetSumWaysRecursive({1,1,1,1,1}, 2), 0);
  EXPECT_EQ(sol.findTargetSumWaysRecursive({1,1,1,1,1}, 1), 10);
  EXPECT_EQ(sol.findTargetSumWaysRecursive({1,1,1,1,1}, 0), 0);
}

TEST_F(PackageProTest, findMaxForm_ok) {
  PackageProSolution sol;
  // "10", "0001", "1", "0"
  EXPECT_EQ(sol.findMaxForm({"10", "0001", "111001", "1", "0"}, 5, 3), 4);
  // "0" "1"
  EXPECT_EQ(sol.findMaxForm({"10", "0", "1"}, 1, 1), 2);

  // all of the input
  EXPECT_EQ(sol.findMaxForm({"1", "0", "1", "0", "1", "0", "1"}, 3, 4), 7);
}

TEST_F(PackageProTest, coinChange_ok) {
  PackageProSolution sol;
  EXPECT_EQ(sol.coinChange({1,2,5}, 1), 1);
  EXPECT_EQ(sol.coinChange({1,2,5}, 2), 1);
  EXPECT_EQ(sol.coinChange({1,2,5}, 3), 2); // 1 2
  EXPECT_EQ(sol.coinChange({1,2,5}, 4), 2); // 2 2
  EXPECT_EQ(sol.coinChange({1,2,5}, 5), 1); // 5
  EXPECT_EQ(sol.coinChange({1,2,5}, 6), 2); // 5 1
  EXPECT_EQ(sol.coinChange({1,2,5}, 7), 2); // 5 2
  EXPECT_EQ(sol.coinChange({1,2,5}, 8), 3); // 5 2 1
  EXPECT_EQ(sol.coinChange({1,2,5}, 9), 3); // 5 2 2
  EXPECT_EQ(sol.coinChange({1,2,5}, 10), 2); // 5 5

  EXPECT_EQ(sol.coinChange({2}, 3), -1);

  EXPECT_EQ(sol.coinChange({186,419,83,408}, 6249), 20);
  EXPECT_EQ(sol.coinChange({419,408,186,83}, 6249), 20);
  EXPECT_EQ(sol.coinChange({2,3}, 7), 3); // 3 2 2
  EXPECT_EQ(sol.coinChange({2,3}, 9), 3);
  EXPECT_EQ(sol.coinChange({3,5,19}, 33), 5); // 19 5 3 3 3
}

TEST_F(PackageProTest, combinationSum4_ok) {
  PackageProSolution sol;
  EXPECT_EQ(sol.combinationSum4({1,2}, 3), 3); // 1 2, 2 1, 1 1 1
  EXPECT_EQ(sol.combinationSum4({1,2,3}, 4), 7);
  EXPECT_EQ(sol.combinationSum4({2,3}, 1), 0);
  EXPECT_EQ(sol.combinationSum4({2,3}, 2), 1);
  EXPECT_EQ(sol.combinationSum4({2,3}, 3), 1);
  EXPECT_EQ(sol.combinationSum4({2,3}, 5), 2);
  EXPECT_EQ(sol.combinationSum4({7}, 0), 1);
  EXPECT_EQ(sol.combinationSum4({7}, 7), 1);
  EXPECT_EQ(sol.combinationSum4({7}, 14), 1);

  EXPECT_EQ(sol.trainWays(1), 1); // 1
  EXPECT_EQ(sol.trainWays(2), 2); // 11 2
  EXPECT_EQ(sol.trainWays(3), 3); // 12 21 111
  EXPECT_EQ(sol.trainWays(4), 5); // 121 211 1111 112 22
}
