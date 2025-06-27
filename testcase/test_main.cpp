#include "Others.h"
#include "gtest/gtest.h"

class MainTest : public testing::Test {

};

TEST_F(MainTest, f42_ok) {
  EXPECT_EQ(f42(), 42);
}

TEST_F(MainTest, nok) {
  EXPECT_TRUE(1 > 2);
}
