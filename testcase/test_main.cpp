#include <cstdio>
#include "Others.h"
#include "gtest/gtest.h"

class QueueTestSmpl3 : public testing::Test {
  void SetUp() override {
  }
};

TEST_F(QueueTestSmpl3, f42_ok) {
  EXPECT_EQ(f42(), 42);
}

TEST_F(QueueTestSmpl3, nok) {
  EXPECT_EQ(1, 2);
  EXPECT_EQ(2, 1);
}

