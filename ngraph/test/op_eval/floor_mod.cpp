// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/floor_mod.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, floor_mod)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{4});
    auto b = make_shared<op::Parameter>(element::f32, Shape{4});
    auto floor_mod = make_shared<op::v1::FloorMod>(a, b);
    auto fun = make_shared<Function>(OutputVector{floor_mod}, ParameterVector{a, b});

    std::vector<float> a_value{5.1, -5.1, 5.1, -5.1};
    std::vector<float> b_value{3.0, 3.0, -3.0, -3.0};
    std::vector<float> expected_result{2.1, 0.9, -0.9, -2.1};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{4}, a_value),
                               make_host_tensor<element::Type_t::f32>(Shape{4}, b_value)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{4});
    auto result_data = read_vector<float>(result);
    for (size_t i = 0; i < expected_result.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

TEST(op_eval, floor_mod_i32)
{
    auto a = make_shared<op::Parameter>(element::i32, Shape{6});
    auto b = make_shared<op::Parameter>(element::i32, Shape{6});
    auto floor_mod = make_shared<op::v1::FloorMod>(a, b);
    auto fun = make_shared<Function>(OutputVector{floor_mod}, ParameterVector{a, b});

    std::vector<int32_t> a_value{-4, 7, 5, 4, -7, 8};
    std::vector<int32_t> b_value{2, -3, 8, -2, 3, 5};
    std::vector<int32_t> expected_result{0, -2,  5,  0,  2,  3};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(Shape{6}, a_value),
                               make_host_tensor<element::Type_t::i32>(Shape{6}, b_value)}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    EXPECT_EQ(result->get_shape(), Shape{6});
    auto result_data = read_vector<int32_t>(result);
    for (size_t i = 0; i < expected_result.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
