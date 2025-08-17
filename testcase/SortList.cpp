#include "gtest/gtest.h"
#include <vector>
using namespace std;


int PartitionSort(int a[], int l, int r) { // [l, r]
  int& pivot = a[r];
  int i = l - 1, j = l;
  for (int j = l; j < r; ++j) {
    if (a[j] < pivot) {
      std::swap(a[++i], a[j]);
    }
  }
  std::swap(a[++i], pivot);
  return i;
}


void QuickSortRecursive (int a[], int N) {
  if (a == nullptr || N < 2) {
    return;
  }
  int mid = PartitionSort(a, 0, N-1);
  // [0, mid)
  // mid
  // [mid+1, N)
  QuickSortRecursive(a, mid);
  QuickSortRecursive(a + mid + 1, N - (mid + 1));
}

void Merge2List(int arr[], int N, int mid) {
  vector<int> temp(N);
  int i = 0, j = mid, k = 0;
  while (i < mid && j < N) {
    if (arr[i] <= arr[j]) { // 保证稳定性
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }
  while (i < mid) {
    temp[k++] = arr[i++];
  }
  while (j < N) {
    temp[k++] = arr[j++];
  }
  memcpy_s(arr, N * sizeof(*arr), temp.data(), temp.size() * sizeof(temp[0]));
}

void MergeSortRecursive(int a[], int N) {
  if (a == nullptr || N < 2) {
    return;
  }
  int mid = N >> 1;
  // [0, mid)
  // [mid, N)
  MergeSortRecursive(a, mid);
  MergeSortRecursive(a + mid, N - mid);
  Merge2List(a, N, mid);
}


// void MergeSortList(int a[], int N) {
//   if (a == nullptr || N < 2) {
//     return;
//   }
//   const int mid = N / 2;
//   MergeSortList(a, mid);//[0, mid]
//   MergeSortList(a+mid+1, N-(mid+1)); //[mid+1, N)
//   Merge(a, mid);
// }

class SortTest : public testing::Test {
  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(SortTest, quicksort_ok) {
  int a[] {5,3,2,1,0};
  constexpr int N = sizeof(a)/sizeof(*a);
  QuickSortRecursive(a, N);
  EXPECT_EQ((vector<int>{a, a+N}), (vector<int>{0,1,2,3,5}));
}


TEST_F(SortTest, mergesort_ok) {
  int a[] {5,3,2,1,0};
  constexpr int N = sizeof(a)/sizeof(*a);
  MergeSortRecursive(a, N);
  EXPECT_EQ((vector<int>{a, a+N}), (vector<int>{0,1,2,3,5}));
}
// TEST_F(SumTest, wrap_ok) {
//   EXPECT_EQ(MySum(INT_MIN, -1), INT_MAX);
// }
