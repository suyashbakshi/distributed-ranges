// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/detail/mdarray_shim.hpp>

#include "xhp-tests.hpp"

#if __GNUC__ == 10 && __GNUC_MINOR__ == 4
// mdspan triggers gcc 10 bugs, skip these tests
#else

using T = int;

// TODO: add tests with ISHMEM backend
class Mdspan : public ::testing::Test {
protected:
  std::size_t xdim = 9, ydim = 5, zdim = 2;
  std::size_t n2d = xdim * ydim, n3d = xdim * ydim * zdim;

  std::array<std::size_t, 2> extents2d = {xdim, ydim};
  std::array<std::size_t, 2> extents2dt = {ydim, xdim};
  std::array<std::size_t, 3> extents3d = {xdim, ydim, zdim};
  std::array<std::size_t, 3> extents3dt = {ydim, zdim, xdim};

  // 2d data with 1d decomposition
  dr::mhp::distribution dist2d_1d = dr::mhp::distribution().granularity(ydim);
  // 3d data with 1d decomposition
  dr::mhp::distribution dist3d_1d =
      dr::mhp::distribution().granularity(ydim * zdim);

  std::array<std::size_t, 2> slice_starts = {1, 1};
  std::array<std::size_t, 2> slice_ends = {3, 3};
};

TEST_F(Mdspan, StaticAssert) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);
  static_assert(rng::forward_range<decltype(mdspan)>);
  static_assert(dr::distributed_range<decltype(mdspan)>);
  auto segments = dr::ranges::segments(mdspan);
  // Begin on a lvalue
  rng::begin(segments);
  // Begin on a rvalue
  // rng::begin(dr::ranges::segments(mdspan));
}

TEST_F(Mdspan, Iterator) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);

  *mdspan.begin() = 17;
  xhp::fence();
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST_F(Mdspan, Mdindex2D) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  std::size_t i = 1, j = 2;
  dmdspan.mdspan()(i, j) = 17;
  xhp::fence();
  EXPECT_EQ(17, dist[i * ydim + j]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j));
}

TEST_F(Mdspan, Mdindex3D) {
  xhp::distributed_vector<T> dist(n3d, dist3d_1d);
  auto dmdspan = xhp::views::mdspan(dist, extents3d);

  std::size_t i = 1, j = 2, k = 0;
  dmdspan.mdspan()(i, j, k) = 17;
  xhp::fence();
  EXPECT_EQ(17, dist[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j, k));
}

TEST_F(Mdspan, Pipe) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = dist | xhp::views::mdspan(extents2d);

  *mdspan.begin() = 17;
  xhp::fence();
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST_F(Mdspan, SegmentExtents) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  // Sum of leading dimension matches original
  std::size_t x = 0;
  for (auto segment : dr::ranges::segments(dmdspan)) {
    auto extents = segment.mdspan().extents();
    x += extents.extent(0);
    // Non leading dimension are not changed
    EXPECT_EQ(extents2d[1], extents.extent(1));
  }
  EXPECT_EQ(extents2d[0], x);
}

TEST_F(Mdspan, Subrange) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto inner = rng::subrange(dist.begin() + ydim, dist.end() - ydim);
  std::array<std::size_t, 2> inner_extents({extents2d[0] - 2, extents2d[1]});
  auto dmdspan = xhp::views::mdspan(inner, inner_extents);

  // Summing up leading dimension size of segments should equal
  // original minus 2 rows
  std::size_t x = 0;
  for (auto segment : dr::ranges::segments(dmdspan)) {
    auto extents = segment.mdspan().extents();
    x += extents.extent(0);
    // Non leading dimension are not changed
    EXPECT_EQ(extents2d[1], extents.extent(1));
  }
  EXPECT_EQ(extents2d[0], x + 2);
}

TEST_F(Mdspan, GridExtents) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  xhp::iota(dist, 100);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);
  auto grid = dmdspan.grid();

  auto x = 0;
  for (std::size_t i = 0; i < grid.extent(0); i++) {
    x += grid(i, 0).mdspan().extent(0);
  }
  EXPECT_EQ(dmdspan.mdspan().extent(0), x);

  auto y = 0;
  for (std::size_t i = 0; i < grid.extent(1); i++) {
    y += grid(0, i).mdspan().extent(1);
  }
  EXPECT_EQ(dmdspan.mdspan().extent(1), y);
}

TEST_F(Mdspan, GridLocalReference) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  xhp::iota(dist, 100);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);
  auto grid = dmdspan.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mhp::fence();
  EXPECT_EQ(99, dist[0]);
}

using Mdarray = Mdspan;

TEST_F(Mdarray, StaticAssert) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  static_assert(rng::forward_range<decltype(mdarray)>);
  static_assert(dr::distributed_range<decltype(mdarray)>);
  static_assert(dr::distributed_mdspan_range<decltype(mdarray)>);
}

TEST_F(Mdarray, Basic) {
  xhp::distributed_mdarray<T, 2> dist(extents2d);
  xhp::iota(dist, 100);

  md::mdarray<T, dr::__detail::md_extents<2>> local(xdim, ydim);
  rng::iota(&local(0, 0), &local(0, 0) + local.size(), 100);

  EXPECT_EQ(dist.mdspan(), local);
}

TEST_F(Mdarray, Iterator) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);

  *mdarray.begin() = 17;
  xhp::fence();
  EXPECT_EQ(17, *mdarray.begin());
  EXPECT_EQ(17, mdarray[0]);
}

auto mdrange_message(auto &mdarray) {
  return fmt::format("Flat: {}\nMdspan:\n{}", mdarray, mdarray.mdspan());
}

TEST_F(Mdarray, Mdindex2D) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::fill(mdarray, 1);

  std::size_t i = 1, j = 2;
  mdarray.mdspan()(i, j) = 17;
  EXPECT_EQ(17, mdarray[i * ydim + j]);
  EXPECT_EQ(17, mdarray.mdspan()(i, j)) << mdrange_message(mdarray);
}

TEST_F(Mdarray, Mdindex3D) {
  xhp::distributed_mdarray<T, 3> mdarray(extents3d);

  std::size_t i = 1, j = 2, k = 0;
  mdarray.mdspan()(i, j, k) = 17;
  xhp::fence();
  EXPECT_EQ(17, mdarray[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, mdarray.mdspan()(i, j, k));
}

TEST_F(Mdarray, GridExtents) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::iota(mdarray, 100);
  auto grid = mdarray.grid();

  auto x = 0;
  for (std::size_t i = 0; i < grid.extent(0); i++) {
    x += grid(i, 0).mdspan().extent(0);
  }
  EXPECT_EQ(mdarray.mdspan().extent(0), x);

  auto y = 0;
  for (std::size_t i = 0; i < grid.extent(1); i++) {
    y += grid(0, i).mdspan().extent(1);
  }
  EXPECT_EQ(mdarray.mdspan().extent(1), y);
}

TEST_F(Mdarray, GridLocalReference) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::iota(mdarray, 100);
  auto grid = mdarray.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mhp::fence();
  EXPECT_EQ(99, mdarray[0]);
}

TEST_F(Mdarray, Halo) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xhp::distributed_mdarray<T, 2> mdarray(extents2d,
                                         xhp::distribution().halo(1));
  dr::mhp::halo(mdarray);
  xhp::iota(mdarray, 100);
  auto grid = mdarray.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mhp::fence();
  EXPECT_EQ(99, mdarray[0]);
}

TEST_F(Mdarray, Enumerate) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  auto e = xhp::views::enumerate(mdarray);
  static_assert(dr::distributed_range<decltype(e)>);
}

TEST_F(Mdarray, Slabs) {
  // local_mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  // leading dimension decomp of 3d array creates slabs
  xhp::distributed_mdarray<T, 3> mdarray(extents3d);
  for (auto slab : dr::mhp::local_mdspans(mdarray)) {
    for (std::size_t i = 0; i < slab.extent(0); i++) {
      for (std::size_t j = 0; j < slab.extent(1); j++) {
        for (std::size_t k = 0; k < slab.extent(2); k++) {
          slab(i, j, k) = 1;
        }
      }
    }
  }
  fence();

  EXPECT_EQ(mdarray.mdspan()(0, 0, 0), 1);
  EXPECT_EQ(
      mdarray.mdspan()(extents3d[0] - 1, extents3d[1] - 1, extents3d[2] - 1),
      1);
}

TEST_F(Mdarray, MdForEach3d) {
  // leading dimension decomp of 3d array creates slabs
  xhp::distributed_mdarray<T, 3> mdarray(extents3d);
  std::vector<T> local(extents3d[0] * extents3d[1] * extents3d[2], 0);
  rng::iota(local, 0);

  auto set = [d1 = extents3d[1], d2 = extents3d[2]](auto index, auto v) {
    auto &[o] = v;
    o = index[0] * d1 * d2 + index[1] * d2 + index[2];
  };
  dr::mhp::for_each(set, mdarray);

  EXPECT_EQ(xhp::views::take(mdarray.view(), local.size()), local)
      << mdrange_message(mdarray);
}

TEST_F(Mdarray, Transpose2D) {
  xhp::distributed_mdarray<double, 2> md_in(extents2d), md_out(extents2dt);
  xhp::iota(md_in, 100);
  xhp::iota(md_out, 200);

  md::mdarray<T, dr::__detail::md_extents<2>> local(extents2dt);
  for (std::size_t i = 0; i < md_out.extent(0); i++) {
    for (std::size_t j = 0; j < md_out.extent(1); j++) {
      local(i, j) = md_in.mdspan()(j, i);
    }
  }

  xhp::transpose(md_in, md_out);
  EXPECT_EQ(md_out.mdspan(), local);
}

TEST_F(Mdarray, Transpose3D) {
  xhp::distributed_mdarray<double, 3> md_in(extents3d), md_out(extents3dt);
  xhp::iota(md_in, 100);
  xhp::iota(md_out, 200);

  md::mdarray<T, dr::__detail::md_extents<3>> local(extents3dt);
  for (std::size_t i = 0; i < md_out.extent(0); i++) {
    for (std::size_t j = 0; j < md_out.extent(1); j++) {
      for (std::size_t k = 0; k < md_out.extent(2); k++) {
        local(i, j, k) = md_in.mdspan()(k, i, j);
      }
    }
  }

  xhp::transpose(md_in, md_out);
  EXPECT_EQ(local, md_out.mdspan()) << fmt::format("md_in\n{}", md_in.mdspan());
}

using Submdspan = Mdspan;

TEST_F(Submdspan, StaticAssert) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  auto submdspan =
      xhp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  static_assert(rng::forward_range<decltype(submdspan)>);
  static_assert(dr::distributed_range<decltype(submdspan)>);
}

TEST_F(Submdspan, Mdindex2D) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::fill(mdarray, 1);
  auto sub = xhp::views::submdspan(mdarray.view(), slice_starts, slice_ends);

  std::size_t i = 1, j = 0;
  sub.mdspan()(i, j) = 17;
  xhp::fence();
  EXPECT_EQ(17, sub.mdspan()(i, j));
  EXPECT_EQ(17, mdarray.mdspan()(slice_starts[0] + i, slice_starts[1] + j));
  EXPECT_EQ(17, mdarray[(i + slice_starts[0]) * ydim + j + slice_starts[1]])
      << mdrange_message(mdarray);
}

TEST_F(Submdspan, GridExtents) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::iota(mdarray, 100);
  auto sub = xhp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  auto grid = sub.grid();
  EXPECT_EQ(slice_ends[0] - slice_starts[0], sub.mdspan().extent(0));
  EXPECT_EQ(slice_ends[1] - slice_starts[1], sub.mdspan().extent(1));

  auto x = 0;
  for (std::size_t i = 0; i < grid.extent(0); i++) {
    x += grid(i, 0).mdspan().extent(0);
  }
  EXPECT_EQ(slice_ends[0] - slice_starts[0], x);
  EXPECT_EQ(slice_ends[0] - slice_starts[0], sub.mdspan().extent(0));

  auto y = 0;
  for (std::size_t i = 0; i < grid.extent(1); i++) {
    y += grid(0, i).mdspan().extent(1);
  }
  EXPECT_EQ(slice_ends[1] - slice_starts[1], y);
  EXPECT_EQ(slice_ends[1] - slice_starts[1], sub.mdspan().extent(1));
}

TEST_F(Submdspan, GridLocalReference) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::iota(mdarray, 100);
  auto sub = xhp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  auto grid = sub.grid();

  std::size_t i = 0, j = 0;
  auto tile = grid(0, 0).mdspan();
  if (tile.extent(0) == 0 || tile.extent(1) == 0) {
    return;
  }
  if (comm_rank == 0) {
    tile(i, j) = 99;
    EXPECT_EQ(99, tile(i, j));
  }
  dr::mhp::fence();

  auto flat_index = (i + slice_starts[0]) * extents2d[1] + slice_starts[1] + j;
  EXPECT_EQ(99, mdarray[flat_index]) << mdrange_message(mdarray);
}

TEST_F(Submdspan, Segments) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::iota(mdarray, 100);
  auto sub = xhp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  auto sspan = sub.mdspan();
  auto sub_segments = dr::ranges::segments(sub);
  using segment_type = rng::range_value_t<decltype(sub_segments)>;

  segment_type first, last;
  bool found_first = false;
  for (auto segment : sub_segments) {
    if (segment.mdspan().extent(0) != 0) {
      if (!found_first) {
        first = segment;
        found_first = true;
      }
      last = segment;
    }
  }

  if (comm_rank == dr::ranges::rank(first)) {
    auto fspan = first.mdspan();
    auto message = fmt::format("Sub:\n{}First:\n{}", sspan, fspan);
    EXPECT_EQ(sspan(0, 0), fspan(0, 0)) << message;
  }

  if (comm_rank == dr::ranges::rank(last)) {
    auto lspan = last.mdspan();
    auto message = fmt::format("Sub:\n{}Last:\n{}", sspan, lspan);
    EXPECT_EQ(sspan(sspan.extent(0) - 1, sspan.extent(1) - 1),
              lspan(lspan.extent(0) - 1, lspan.extent(1) - 1))
        << message;
  }

  dr::mhp::barrier();
}

using MdForeach = Mdspan;

TEST_F(MdForeach, 2ops) {
  xhp::distributed_mdarray<T, 2> a(extents2d);
  xhp::distributed_mdarray<T, 2> b(extents2d);
  auto mda = a.mdspan();
  auto mdb = b.mdspan();
  xhp::iota(a, 100);
  xhp::iota(b, 200);
  auto copy_op = [](auto v) {
    auto &[in, out] = v;
    out = in;
  };

  xhp::for_each(copy_op, a, b);
  EXPECT_EQ(mda(0, 0), mdb(0, 0));
  EXPECT_EQ(mda(xdim - 1, ydim - 1), mdb(xdim - 1, ydim - 1));
}

TEST_F(MdForeach, 3ops) {
  xhp::distributed_mdarray<T, 2> a(extents2d);
  xhp::distributed_mdarray<T, 2> b(extents2d);
  xhp::distributed_mdarray<T, 2> c(extents2d);
  xhp::iota(a, 100);
  xhp::iota(b, 200);
  xhp::iota(c, 200);
  auto copy_op = [](auto v) {
    auto [in1, in2, out] = v;
    out = in1 + in2;
  };

  xhp::for_each(copy_op, a, b, c);
  EXPECT_EQ(a.mdspan()(2, 2) + b.mdspan()(2, 2), c.mdspan()(2, 2))
      << fmt::format("A:\n{}\nB:\n{}\nC:\n{}", a.mdspan(), b.mdspan(),
                     c.mdspan());
}

TEST_F(MdForeach, Indexed) {
  xhp::distributed_mdarray<T, 2> dist(extents2d);
  auto op = [l = ydim](auto index, auto v) {
    auto &[o] = v;
    o = index[0] * l + index[1];
  };

  xhp::for_each(op, dist);
  for (std::size_t i = 0; i < xdim; i++) {
    for (std::size_t j = 0; j < ydim; j++) {
      EXPECT_EQ(dist.mdspan()(i, j), i * ydim + j)
          << fmt::format("i: {} j: {}\n", i, j);
    }
  }
}

using MdStencilForeach = Mdspan;

TEST_F(MdStencilForeach, 2ops) {
  xhp::distributed_mdarray<T, 2> a(extents2d);
  xhp::distributed_mdarray<T, 2> b(extents2d);
  xhp::iota(a, 100);
  xhp::iota(b, 200);
  auto mda = a.mdspan();
  auto mdb = b.mdspan();
  auto copy_op = [](auto v) {
    auto [in, out] = v;
    out(0, 0) = in(0, 0);
  };

  xhp::stencil_for_each(copy_op, a, b);
  EXPECT_EQ(mda(0, 0), mdb(0, 0));
  EXPECT_EQ(mda(2, 2), mdb(2, 2));
  EXPECT_EQ(mda(xdim - 1, ydim - 1), mdb(xdim - 1, ydim - 1));
}

TEST_F(MdStencilForeach, 3ops) {
  xhp::distributed_mdarray<T, 2> a(extents2d);
  xhp::distributed_mdarray<T, 2> b(extents2d);
  xhp::distributed_mdarray<T, 2> c(extents2d);
  xhp::iota(a, 100);
  xhp::iota(b, 200);
  xhp::iota(c, 300);
  auto copy_op = [](auto v) {
    auto [in1, in2, out] = v;
    out(0, 0) = in1(0, 0) + in2(0, 0);
  };

  xhp::stencil_for_each(copy_op, a, b, c);
  EXPECT_EQ(a.mdspan()(2, 2) + b.mdspan()(2, 2), c.mdspan()(2, 2));
}

using MdspanUtil = Mdspan;

TEST_F(MdspanUtil, Pack) {
  std::vector<T> a(xdim * ydim);
  std::vector<T> b(xdim * ydim);
  rng::iota(a, 100);
  rng::iota(b, 100);

  dr::__detail::mdspan_copy(md::mdspan(a.data(), extents2d), b.begin());
  EXPECT_EQ(a, b);
}

TEST_F(MdspanUtil, UnPack) {
  std::vector<T> a(xdim * ydim);
  std::vector<T> b(xdim * ydim);
  rng::iota(a, 100);
  rng::iota(b, 100);

  dr::__detail::mdspan_copy(a.begin(), md::mdspan(b.data(), extents2d));
  EXPECT_EQ(a, b);
}

TEST_F(MdspanUtil, Copy) {
  std::vector<T> a(xdim * ydim);
  std::vector<T> b(xdim * ydim);
  rng::iota(a, 100);
  rng::iota(b, 100);

  dr::__detail::mdspan_copy(md::mdspan(a.data(), extents2d),
                            md::mdspan(b.data(), extents2d));
  EXPECT_EQ(a, b);
}

TEST_F(MdspanUtil, Transpose2D) {
  std::vector<T> a(xdim * ydim);
  std::vector<T> b(xdim * ydim);
  std::vector<T> c(xdim * ydim);
  rng::iota(a, 100);
  rng::iota(b, 200);
  rng::iota(c, 300);
  md::mdspan mda(a.data(), extents2d);
  md::mdspan mdc(c.data(), extents2dt);

  md::mdarray<T, dr::__detail::md_extents<2>> ref(extents2dt);
  std::vector<T> ref_packed(xdim * ydim);
  T *rp = ref_packed.data();
  for (std::size_t i = 0; i < ref.extent(0); i++) {
    for (std::size_t j = 0; j < ref.extent(1); j++) {
      ref(i, j) = mda(j, i);
      *rp++ = ref(i, j);
    }
  }

  // Transpose view
  dr::__detail::mdtranspose<decltype(mda), 1, 0> mdat(mda);
  EXPECT_EQ(ref, mdat);

  // Transpose pack
  dr::__detail::mdspan_copy(mdat, b.begin());
  EXPECT_EQ(ref_packed, b);

  // Transpose copy
  dr::__detail::mdspan_copy(mdat, mdc);
  EXPECT_EQ(mdat, mdc);
}

TEST_F(MdspanUtil, Transpose3D) {
  md::mdarray<T, dr::__detail::md_extents<3>> md(extents3d),
      mdt_ref(extents3dt);
  T *base = &md(0, 0, 0);
  rng::iota(rng::subrange(base, base + md.size()), 100);

  std::vector<T> ref_packed(md.size()), packed(md.size());

  T *rp = ref_packed.data();
  for (std::size_t i = 0; i < mdt_ref.extent(0); i++) {
    for (std::size_t j = 0; j < mdt_ref.extent(1); j++) {
      for (std::size_t k = 0; k < mdt_ref.extent(2); k++) {
        mdt_ref(i, j, k) = md(k, i, j);
        *rp++ = mdt_ref(i, j, k);
      }
    }
  }

  // Transpose view
  auto mdspan = md.to_mdspan();
  dr::__detail::mdtranspose<decltype(mdspan), 2, 0, 1> mdt(mdspan);
  EXPECT_EQ(mdt_ref.to_mdspan(), mdt);

  // Transpose pack
  dr::__detail::mdspan_copy(mdt, packed.begin());
  EXPECT_EQ(ref_packed, packed);
}

#endif // Skip for gcc 10.4
