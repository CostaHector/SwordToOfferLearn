#include "gtest/gtest.h"

int MySum(int a, int b) {
  return a + b;
}

class SumTest : public testing::Test {
  void SetUp() override {
  }
};

TEST_F(SumTest, positive_negative_ok) {
  EXPECT_EQ(MySum(1, -1), 0);
}

TEST_F(SumTest, wrap_ok) {
  EXPECT_EQ(MySum(INT_MIN, -1), INT_MAX);
}
