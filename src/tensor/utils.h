#pragma once
#include <vector>
#include <functional>
#include <cstdint>

// ============================================================
// for_each_index
//
// Visits every valid multi-index of a given shape,
// in row-major (C) order, and calls fn with that index.
//
// This is the n-dimensional equivalent of a nested for loop.
// Instead of writing:
//
//   for (int r = 0; r < rows; r++)
//     for (int c = 0; c < cols; c++)
//       fn({r, c});
//
// You write:
//
//   for_each_index({rows, cols}, fn);
//
// And it works for any number of dimensions.
//
// Examples:
//   shape {4}       → visits {0}, {1}, {2}, {3}
//   shape {2,3}     → visits {0,0},{0,1},{0,2},{1,0},{1,1},{1,2}
//   shape {2,2,2}   → visits {0,0,0},{0,0,1},{0,1,0},...,{1,1,1}
// ============================================================
inline void for_each_index(
    const std::vector<int64_t>& shape,
    std::function<void(const std::vector<int64_t>&)> fn)
{
    int ndim = (int)shape.size();

    // edge case: scalar tensor has no dimensions
    // visit once with an empty index
    if (ndim == 0) {
        fn({});
        return;
    }

    // edge case: if any dimension is zero, there are no elements
    // nothing to visit
    for (int d = 0; d < ndim; ++d) {
        if (shape[d] == 0) return;
    }

    // start at the first valid index: {0, 0, ..., 0}
    std::vector<int64_t> idx(ndim, 0);

    // compute total number of elements to visit
    // = product of all dimensions
    int64_t total = 1;
    for (int64_t s : shape) total *= s;

    for (int64_t n = 0; n < total; ++n) {

        // call the function with the current index
        fn(idx);

        // advance to the next index in row-major order
        // this mirrors how a nested loop's innermost counter
        // increments first, carrying over to outer dimensions
        //
        // example with shape {2,3}:
        //   {0,0} → increment dim 1 → {0,1}
        //   {0,1} → increment dim 1 → {0,2}
        //   {0,2} → dim 1 overflows → reset to 0, carry to dim 0
        //        → increment dim 0 → {1,0}
        //   {1,0} → increment dim 1 → {1,1}
        //   ...
        for (int d = ndim - 1; d >= 0; --d) {

            ++idx[d];

            if (idx[d] < shape[d]) {
                // no overflow — stop carrying
                break;
            }

            // overflow — reset this dimension and carry to the next
            idx[d] = 0;

            // if d == 0 and we're still here, we've wrapped around
            // past the last element — loop will exit naturally since
            // n == total - 1 at that point
        }
    }
}