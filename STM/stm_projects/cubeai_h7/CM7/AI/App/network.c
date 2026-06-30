/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-10T22:19:20+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x5fd20703d5f909978c4a11f12528c751"
#define _STAI_NETWORK_DATETIME            "2026-06-10T22:19:20+0800"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 3, 96, 96),
      STAI_DECLARE_ARRAY(float, 1, 0.007843106053769588f),
      STAI_DECLARE_ARRAY(int16_t, 1, 0)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 100),
      STAI_DECLARE_ARRAY(float, 1, 1.3354321708902717e-05f),
      STAI_DECLARE_ARRAY(int16_t, 1, 14)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 63268),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 53232),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843106053769588f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_Transpose_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843106053769588f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stem_conv1_stem_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split1_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split2_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split3_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split4_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split5_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split6_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split7_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split8_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split9_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0026632575318217278f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004274711012840271f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004869819153100252f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006499545648694038f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004104751627892256f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0047948118299245834f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0051765721291303635f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004307139664888382f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004592200741171837f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005467839539051056f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006499545648694038f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split10_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split11_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split12_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split13_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split14_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split15_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split16_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split17_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split18_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split19_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004460128955543041f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003856317140161991f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004413940478116274f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00526389479637146f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005249801557511091f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0037528062239289284f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004686704371124506f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004232417792081833f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004691829439252615f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005343812983483076f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005343812983483076f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split20_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split21_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split22_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___stem_conv1_stem_Conv_output_0_split23_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01088507566601038f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005204447079449892f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004810446407645941f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0047269705682992935f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004641928244382143f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005204447079449892f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006499545648694038f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001367607619613409f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_0_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006499545648694038f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_0_Transpose_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006499545648694038f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00047203159192577004f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005766698159277439f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_1_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005766698159277439f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_0__layers_1_Transpose_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005766698159277439f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.87527353875339e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001844168291427195f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_0_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001844168291427195f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_0_Transpose_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001844168291427195f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.599483509082347e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00013865303480997682f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_1_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00013865303480997682f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_1_Transpose_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00013865303480997682f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.370134411146864e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_2_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_1__layers_2_Transpose_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00010105739784194157f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.631847933633253e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.459926513954997e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.460085129830986e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.458923886995763e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.461980516789481e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.459777356823906e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.459783177589998e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.461582521907985e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.461899753659964e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00010105739784194157f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012253265595063567f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #97 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.404182492289692e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #98 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.717770833754912e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #99 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.404182492289692e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #100 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00010105739784194157f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #101 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00019421213073655963f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #102 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_0_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00019421213073655963f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #103 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_0_Transpose_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00019421213073655963f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #104 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.426796658663079e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #105 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.565056097926572e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #106 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_1_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.565056097926572e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #107 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_choice_blocks_2__layers_1_Transpose_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.565056097926572e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #108 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_global_pooling_GlobalAveragePool_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.17522847885266e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #109 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(output_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.3354321708902717e-05f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #110 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(output_QuantizeLinear_Input_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 100,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00019223647541366518f, 0.00020763350767083466f, 0.0003008080238942057f, 0.00018630956765264273f, 0.00022261113917920738f, 0.00019103231898043305f, 0.00020235631382092834f, 0.0002091199712594971f, 0.00025332270888611674f, 0.00026996928500011563f, 0.0002404254482826218f, 0.0001854548609117046f, 0.00019476524903438985f, 0.00017403899983037263f, 0.00024845098960213363f, 0.00021175044821575284f, 0.00024796437355689704f, 0.00020026344282086939f, 0.0002688096137717366f, 0.00022002722835168242f, 0.00020065192074980587f, 0.00019929905829485506f, 0.00018416490638628602f, 0.00026276748394593596f, 0.00025201388052664697f, 0.0002148194907931611f, 0.0002376541233388707f, 0.00031507876701653004f, 0.0002044720167759806f, 0.00021169301180634648f, 0.00025057437596842647f, 0.0002249773679068312f, 0.00019175087800249457f, 0.00024771541939117014f, 0.0002119200798915699f, 0.000246829615207389f, 0.00017425879195798188f, 0.00027053209487348795f, 0.0002182242023991421f, 0.00023741395852994174f, 0.00018578037270344794f, 0.0002498071116860956f, 0.00020773769938386977f, 0.0002174700057366863f, 0.0002194738481193781f, 0.00021249368728604168f, 0.00022422787151299417f, 0.0001737291313474998f, 0.00023191094805952162f, 0.00018160490435548127f, 0.00030495855025947094f, 0.00020446594862733036f, 0.00020362602663226426f, 0.00018255402392242104f, 0.00023996259551495314f, 0.00020864115504082292f, 0.0002737595059443265f, 0.0002000089589273557f, 0.00023896429047454149f, 0.000204950338229537f, 0.00020445050904527307f, 0.00021503824973478913f, 0.00015744316624477506f, 0.00017640630539972335f, 0.00025761386496014893f, 0.0002098180411849171f, 0.00021236840984784067f, 0.0002327958500245586f, 0.00020954827778041363f, 0.00022456870647147298f, 0.00018737153732217848f, 0.00021296387421898544f, 0.00020630819199141115f, 0.0002014577476074919f, 0.00024214654695242643f, 0.00021234981250017881f, 0.00019552867161110044f, 0.00022246103617362678f, 0.000262681016465649f, 0.0002081597049254924f, 0.00019315005920361727f, 0.00017094095528591424f, 0.0001509045105194673f, 0.0003023490135092288f, 0.00022420103778131306f, 0.00019346363842487335f, 0.000202710521989502f, 0.00018872597138397396f, 0.00018457227270118892f, 0.00021072918025311083f, 0.0001998203806579113f, 0.00022868273663334548f, 0.00016277712711598724f, 0.00019064988009631634f, 0.0001934258034452796f, 0.00021550535166170448f, 0.0001612215710338205f, 0.00023955866345204413f, 0.00023828793200664222f, 0.0002975591050926596f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 27648, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_Transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27649, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _stem_conv1_stem_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split1_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split2_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split3_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split4_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split5_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split6_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split7_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split8_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split9_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2880, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split10_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split11_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split12_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split13_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split14_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split15_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split16_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split17_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split18_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split19_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2880, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split20_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split21_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split22_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split23_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5760, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8064, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5760, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  _global_pooling_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 100, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 100, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 596, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_Transpose_output, AI_STATIC,
  373, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 96, 96), AI_STRIDE_INIT(4, 1, 1, 3, 288),
  1, &input_Transpose_output_array, &input_Transpose_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  374, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 96, 3), AI_STRIDE_INIT(4, 1, 1, 96, 9216),
  1, &input_output_array, &input_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _stem_conv1_stem_Conv_output_0_output, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 48), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &_stem_conv1_stem_Conv_output_0_output_array, &_stem_conv1_stem_Conv_output_0_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split0_out_output, AI_STATIC,
  272, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 3), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split0_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split0_out_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split1_out_output, AI_STATIC,
  283, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split1_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split1_out_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split2_out_output, AI_STATIC,
  288, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split2_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split2_out_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split3_out_output, AI_STATIC,
  289, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split3_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split3_out_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split4_out_output, AI_STATIC,
  290, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split4_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split4_out_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split5_out_output, AI_STATIC,
  291, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split5_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split5_out_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split6_out_output, AI_STATIC,
  292, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split6_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split6_out_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split7_out_output, AI_STATIC,
  293, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split7_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split7_out_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split8_out_output, AI_STATIC,
  294, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split8_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split8_out_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split9_out_output, AI_STATIC,
  295, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split9_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split9_out_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output, AI_STATIC,
  196, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 10), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output, AI_STATIC,
  200, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output, AI_STATIC,
  234, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output, AI_STATIC,
  249, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output, AI_STATIC,
  252, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output, AI_STATIC,
  255, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output, AI_STATIC,
  258, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output, AI_STATIC,
  261, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output, AI_STATIC,
  264, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output, AI_STATIC,
  267, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output, AI_STATIC,
  270, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split10_out_output, AI_STATIC,
  273, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split10_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split10_out_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split11_out_output, AI_STATIC,
  274, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split11_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split11_out_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split12_out_output, AI_STATIC,
  275, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split12_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split12_out_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split13_out_output, AI_STATIC,
  276, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split13_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split13_out_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split14_out_output, AI_STATIC,
  277, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split14_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split14_out_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split15_out_output, AI_STATIC,
  278, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split15_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split15_out_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split16_out_output, AI_STATIC,
  279, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split16_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split16_out_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split17_out_output, AI_STATIC,
  280, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split17_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split17_out_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split18_out_output, AI_STATIC,
  281, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split18_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split18_out_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split19_out_output, AI_STATIC,
  282, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split19_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split19_out_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output, AI_STATIC,
  197, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 10), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output, AI_STATIC,
  204, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output, AI_STATIC,
  207, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output, AI_STATIC,
  210, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output, AI_STATIC,
  213, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output, AI_STATIC,
  216, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output, AI_STATIC,
  219, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output, AI_STATIC,
  222, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output, AI_STATIC,
  225, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output, AI_STATIC,
  228, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output, AI_STATIC,
  231, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split20_out_output, AI_STATIC,
  284, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split20_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split20_out_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split21_out_output, AI_STATIC,
  285, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split21_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split21_out_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split22_out_output, AI_STATIC,
  286, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 5), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split22_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split22_out_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split23_out_output, AI_STATIC,
  287, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 4), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &g2_5___stem_conv1_stem_Conv_output_0_split23_out_output_array, &g2_5___stem_conv1_stem_Conv_output_0_split23_out_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output, AI_STATIC,
  198, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 4), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output, AI_STATIC,
  237, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output, AI_STATIC,
  240, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output, AI_STATIC,
  243, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output, AI_STATIC,
  246, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 1), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output_array, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 24), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array, &_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Concat_output_0_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_0__layers_0_Concat_output_0_output_array, &_choice_blocks_0__layers_0_Concat_output_0_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 24), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &_choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, &_choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Concat_output_0_output0, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(5, 1, 12, 24, 24, 2), AI_STRIDE_INIT(5, 1, 1, 24, 576, 12),
  1, &_choice_blocks_0__layers_0_Concat_output_0_output_array, &_choice_blocks_0__layers_0_Concat_output_0_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Transpose_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(5, 1, 2, 24, 24, 12), AI_STRIDE_INIT(5, 1, 1, 24, 576, 2),
  1, &_choice_blocks_0__layers_0_Transpose_output_0_output_array, &_choice_blocks_0__layers_0_Transpose_output_0_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Concat_output_0_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_0__layers_1_Concat_output_0_output_array, &_choice_blocks_0__layers_1_Concat_output_0_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 24), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &_choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, &_choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 24, 24), AI_STRIDE_INIT(4, 1, 1, 12, 288),
  1, &_choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, &_choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Concat_output_0_output0, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(5, 1, 12, 24, 24, 2), AI_STRIDE_INIT(5, 1, 1, 24, 576, 12),
  1, &_choice_blocks_0__layers_1_Concat_output_0_output_array, &_choice_blocks_0__layers_1_Concat_output_0_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Transpose_output_0_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(5, 1, 2, 24, 24, 12), AI_STRIDE_INIT(5, 1, 1, 24, 576, 2),
  1, &_choice_blocks_0__layers_1_Transpose_output_0_output_array, &_choice_blocks_0__layers_1_Transpose_output_0_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Concat_output_0_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 24), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &_choice_blocks_1__layers_0_Concat_output_0_output_array, &_choice_blocks_1__layers_0_Concat_output_0_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, &_choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, &_choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Concat_output_0_output0, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(5, 1, 24, 24, 24, 2), AI_STRIDE_INIT(5, 1, 1, 48, 1152, 24),
  1, &_choice_blocks_1__layers_0_Concat_output_0_output_array, &_choice_blocks_1__layers_0_Concat_output_0_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Transpose_output_0_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(5, 1, 2, 24, 24, 24), AI_STRIDE_INIT(5, 1, 1, 48, 1152, 2),
  1, &_choice_blocks_1__layers_0_Transpose_output_0_output_array, &_choice_blocks_1__layers_0_Transpose_output_0_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Concat_output_0_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 24), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &_choice_blocks_1__layers_1_Concat_output_0_output_array, &_choice_blocks_1__layers_1_Concat_output_0_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, &_choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, &_choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Concat_output_0_output0, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(5, 1, 24, 24, 24, 2), AI_STRIDE_INIT(5, 1, 1, 48, 1152, 24),
  1, &_choice_blocks_1__layers_1_Concat_output_0_output_array, &_choice_blocks_1__layers_1_Concat_output_0_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Transpose_output_0_output, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(5, 1, 2, 24, 24, 24), AI_STRIDE_INIT(5, 1, 1, 48, 1152, 2),
  1, &_choice_blocks_1__layers_1_Transpose_output_0_output_array, &_choice_blocks_1__layers_1_Transpose_output_0_output_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Concat_output_0_output, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 24), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &_choice_blocks_1__layers_2_Concat_output_0_output_array, &_choice_blocks_1__layers_2_Concat_output_0_output_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output, AI_STATIC,
  84, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, &_choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 24, 24), AI_STRIDE_INIT(4, 1, 1, 24, 576),
  1, &_choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, &_choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Concat_output_0_output0, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(5, 1, 24, 24, 24, 2), AI_STRIDE_INIT(5, 1, 1, 48, 1152, 24),
  1, &_choice_blocks_1__layers_2_Concat_output_0_output_array, &_choice_blocks_1__layers_2_Concat_output_0_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Transpose_output_0_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(5, 1, 2, 24, 24, 24), AI_STRIDE_INIT(5, 1, 1, 48, 1152, 2),
  1, &_choice_blocks_1__layers_2_Transpose_output_0_output_array, &_choice_blocks_1__layers_2_Transpose_output_0_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Transpose_output_0_output0, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 24), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &_choice_blocks_1__layers_2_Transpose_output_0_output_array, &_choice_blocks_1__layers_2_Transpose_output_0_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output, AI_STATIC,
  296, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 4), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output_array_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output, AI_STATIC,
  299, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 6), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output, AI_STATIC,
  300, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output_array_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output, AI_STATIC,
  301, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output, AI_STATIC,
  302, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output_array_intq)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output, AI_STATIC,
  303, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output, AI_STATIC,
  304, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output_array_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output, AI_STATIC,
  305, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output, AI_STATIC,
  306, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output_array_intq)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output, AI_STATIC,
  307, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output_array_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output, AI_STATIC,
  334, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 10), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output, AI_STATIC,
  337, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output, AI_STATIC,
  347, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output, AI_STATIC,
  350, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array_intq)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output, AI_STATIC,
  353, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output, AI_STATIC,
  356, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array_intq)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output, AI_STATIC,
  359, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output, AI_STATIC,
  362, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array_intq)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output, AI_STATIC,
  365, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output, AI_STATIC,
  368, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array_intq)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output, AI_STATIC,
  371, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output, AI_STATIC,
  297, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 7), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output_array_intq)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output, AI_STATIC,
  298, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 24, 5), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output_array, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output, AI_STATIC,
  335, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 2), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output, AI_STATIC,
  341, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output, AI_STATIC,
  344, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 1), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 12), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array, &_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array_intq)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Concat_output_0_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 12, 12), AI_STRIDE_INIT(4, 1, 1, 96, 1152),
  1, &_choice_blocks_2__layers_0_Concat_output_0_output_array, &_choice_blocks_2__layers_0_Concat_output_0_output_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output, AI_STATIC,
  114, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 12), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &_choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, &_choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Concat_output_0_output0, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(5, 1, 48, 12, 12, 2), AI_STRIDE_INIT(5, 1, 1, 96, 1152, 48),
  1, &_choice_blocks_2__layers_0_Concat_output_0_output_array, &_choice_blocks_2__layers_0_Concat_output_0_output_array_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Transpose_output_0_output, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(5, 1, 2, 12, 12, 48), AI_STRIDE_INIT(5, 1, 1, 96, 1152, 2),
  1, &_choice_blocks_2__layers_0_Transpose_output_0_output_array, &_choice_blocks_2__layers_0_Transpose_output_0_output_array_intq)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Concat_output_0_output, AI_STATIC,
  117, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 12, 12), AI_STRIDE_INIT(4, 1, 1, 96, 1152),
  1, &_choice_blocks_2__layers_1_Concat_output_0_output_array, &_choice_blocks_2__layers_1_Concat_output_0_output_array_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output, AI_STATIC,
  122, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 12), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &_choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array, &_choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output, AI_STATIC,
  134, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 12, 12), AI_STRIDE_INIT(4, 1, 1, 48, 576),
  1, &_choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array, &_choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Concat_output_0_output0, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(5, 1, 48, 12, 12, 2), AI_STRIDE_INIT(5, 1, 1, 96, 1152, 48),
  1, &_choice_blocks_2__layers_1_Concat_output_0_output_array, &_choice_blocks_2__layers_1_Concat_output_0_output_array_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Transpose_output_0_output, AI_STATIC,
  119, 0x1,
  AI_SHAPE_INIT(5, 1, 2, 12, 12, 48), AI_STRIDE_INIT(5, 1, 1, 96, 1152, 2),
  1, &_choice_blocks_2__layers_1_Transpose_output_0_output_array, &_choice_blocks_2__layers_1_Transpose_output_0_output_array_intq)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Transpose_output_0_output0, AI_STATIC,
  120, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 12, 12), AI_STRIDE_INIT(4, 1, 1, 96, 1152),
  1, &_choice_blocks_2__layers_1_Transpose_output_0_output_array, &_choice_blocks_2__layers_1_Transpose_output_0_output_array_intq)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  _global_pooling_GlobalAveragePool_output_0_output, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_global_pooling_GlobalAveragePool_output_0_output_array, &_global_pooling_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_bias, AI_STATIC,
  375, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &output_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_output, AI_STATIC,
  376, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &output_QuantizeLinear_Input_output_array, &output_QuantizeLinear_Input_output_array_intq)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_scratch0, AI_STATIC,
  377, 0x0,
  AI_SHAPE_INIT(4, 1, 596, 1, 1), AI_STRIDE_INIT(4, 2, 2, 1192, 1192),
  1, &output_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_weights, AI_STATIC,
  378, 0x1,
  AI_SHAPE_INIT(4, 96, 100, 1, 1), AI_STRIDE_INIT(4, 1, 96, 9600, 9600),
  1, &output_QuantizeLinear_Input_weights_array, &output_QuantizeLinear_Input_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_Transpose_chain,
  NULL, &input_Transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split0_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split0_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split0_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split0_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split0_out_starts_data[] = { 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split0_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split0_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split0_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split0_out_ends_data[] = { 3, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split0_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split0_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split0_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split0_out_layer, 166,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split0_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split0_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split0_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split0_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split0_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split1_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split1_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split1_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split1_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split1_out_starts_data[] = { 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split1_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split1_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split1_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split1_out_ends_data[] = { 5, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split1_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split1_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split1_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split1_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split1_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split1_out_layer, 165,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split1_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split1_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split1_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split1_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split1_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split2_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split2_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split2_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split2_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split2_out_starts_data[] = { 2, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split2_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split2_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split2_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split2_out_ends_data[] = { 7, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split2_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split2_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split2_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split2_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split2_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split2_out_layer, 164,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split2_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split2_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split2_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split2_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split2_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split3_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split3_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split3_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split3_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split3_out_starts_data[] = { 4, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split3_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split3_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split3_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split3_out_ends_data[] = { 9, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split3_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split3_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split3_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split3_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split3_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split3_out_layer, 163,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split3_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split3_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split3_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split3_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split3_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split4_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split4_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split4_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split4_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split4_out_starts_data[] = { 6, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split4_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split4_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split4_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split4_out_ends_data[] = { 11, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split4_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split4_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split4_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split4_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split4_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split4_out_layer, 162,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split4_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split4_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split4_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split4_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split4_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split5_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split5_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split5_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split5_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split5_out_starts_data[] = { 8, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split5_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split5_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split5_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split5_out_ends_data[] = { 13, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split5_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split5_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split5_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split5_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split5_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split5_out_layer, 161,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split5_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split5_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split5_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split5_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split5_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split6_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split6_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split6_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split6_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split6_out_starts_data[] = { 10, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split6_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split6_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split6_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split6_out_ends_data[] = { 15, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split6_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split6_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split6_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split6_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split6_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split6_out_layer, 160,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split6_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split6_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split6_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split6_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split6_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split7_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split7_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split7_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split7_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split7_out_starts_data[] = { 12, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split7_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split7_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split7_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split7_out_ends_data[] = { 17, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split7_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split7_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split7_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split7_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split7_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split7_out_layer, 159,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split7_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split7_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split7_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split7_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split7_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split8_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split8_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split8_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split8_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split8_out_starts_data[] = { 14, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split8_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split8_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split8_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split8_out_ends_data[] = { 19, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split8_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split8_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split8_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split8_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split8_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split8_out_layer, 158,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split8_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split8_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split8_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split8_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split8_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split9_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split9_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split9_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split9_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split9_out_starts_data[] = { 16, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split9_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split9_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split9_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split9_out_ends_data[] = { 21, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split9_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split9_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split9_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split9_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split9_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split9_out_layer, 157,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split9_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split9_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split9_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split9_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split9_out_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 10, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_layer, 370,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_chain,
  NULL, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split10_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split10_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split10_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split10_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split10_out_starts_data[] = { 18, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split10_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split10_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split10_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split10_out_ends_data[] = { 23, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split10_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split10_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split10_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split10_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split10_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split10_out_layer, 156,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split10_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split10_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split10_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split10_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split10_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split11_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split11_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split11_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split11_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split11_out_starts_data[] = { 20, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split11_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split11_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split11_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split11_out_ends_data[] = { 25, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split11_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split11_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split11_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split11_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split11_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split11_out_layer, 155,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split11_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split11_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split11_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split11_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split11_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split12_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split12_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split12_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split12_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split12_out_starts_data[] = { 22, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split12_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split12_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split12_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split12_out_ends_data[] = { 27, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split12_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split12_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split12_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split12_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split12_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split12_out_layer, 154,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split12_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split12_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split12_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split12_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split12_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split13_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split13_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split13_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split13_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split13_out_starts_data[] = { 24, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split13_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split13_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split13_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split13_out_ends_data[] = { 29, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split13_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split13_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split13_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split13_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split13_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split13_out_layer, 153,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split13_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split13_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split13_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split13_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split13_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split14_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split14_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split14_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split14_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split14_out_starts_data[] = { 26, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split14_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split14_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split14_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split14_out_ends_data[] = { 31, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split14_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split14_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split14_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split14_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split14_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split14_out_layer, 152,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split14_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split14_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split14_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split14_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split14_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split15_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split15_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split15_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split15_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split15_out_starts_data[] = { 28, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split15_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split15_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split15_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split15_out_ends_data[] = { 33, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split15_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split15_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split15_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split15_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split15_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split15_out_layer, 151,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split15_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split15_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split15_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split15_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split15_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split16_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split16_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split16_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split16_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split16_out_starts_data[] = { 30, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split16_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split16_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split16_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split16_out_ends_data[] = { 35, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split16_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split16_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split16_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split16_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split16_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split16_out_layer, 150,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split16_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split16_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split16_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split16_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split16_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split17_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split17_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split17_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split17_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split17_out_starts_data[] = { 32, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split17_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split17_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split17_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split17_out_ends_data[] = { 37, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split17_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split17_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split17_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split17_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split17_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split17_out_layer, 149,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split17_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split17_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split17_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split17_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split17_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split18_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split18_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split18_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split18_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split18_out_starts_data[] = { 34, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split18_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split18_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split18_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split18_out_ends_data[] = { 39, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split18_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split18_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split18_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split18_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split18_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split18_out_layer, 148,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split18_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split18_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split18_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split18_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split18_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split19_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split19_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split19_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split19_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split19_out_starts_data[] = { 36, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split19_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split19_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split19_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split19_out_ends_data[] = { 41, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split19_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split19_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split19_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split19_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split19_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split19_out_layer, 147,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split19_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split19_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split19_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split19_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split19_out_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 10, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_layer, 369,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_chain,
  NULL, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split20_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split20_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split20_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split20_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split20_out_starts_data[] = { 38, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split20_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split20_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split20_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split20_out_ends_data[] = { 43, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split20_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split20_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split20_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split20_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split20_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split20_out_layer, 146,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split20_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split20_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split20_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split20_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split20_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split21_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split21_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split21_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split21_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split21_out_starts_data[] = { 40, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split21_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split21_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split21_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split21_out_ends_data[] = { 45, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split21_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split21_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split21_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split21_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split21_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split21_out_layer, 145,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split21_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split21_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split21_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split21_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split21_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split22_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split22_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split22_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split22_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split22_out_starts_data[] = { 42, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split22_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split22_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split22_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split22_out_ends_data[] = { 47, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split22_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split22_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split22_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split22_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split22_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split22_out_layer, 144,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split22_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split22_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split22_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split22_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split22_out_ends, 
)


AI_STATIC_CONST ai_u8 g2_5___stem_conv1_stem_Conv_output_0_split23_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split23_out_axes, AI_ARRAY_FORMAT_U8,
    g2_5___stem_conv1_stem_Conv_output_0_split23_out_axes_data, g2_5___stem_conv1_stem_Conv_output_0_split23_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split23_out_starts_data[] = { 44, 0 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split23_out_starts, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split23_out_starts_data, g2_5___stem_conv1_stem_Conv_output_0_split23_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g2_5___stem_conv1_stem_Conv_output_0_split23_out_ends_data[] = { 48, 48 };
AI_ARRAY_OBJ_DECLARE(
    g2_5___stem_conv1_stem_Conv_output_0_split23_out_ends, AI_ARRAY_FORMAT_S16,
    g2_5___stem_conv1_stem_Conv_output_0_split23_out_ends_data, g2_5___stem_conv1_stem_Conv_output_0_split23_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split23_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_conv1_stem_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___stem_conv1_stem_Conv_output_0_split23_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___stem_conv1_stem_Conv_output_0_split23_out_layer, 143,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g2_5___stem_conv1_stem_Conv_output_0_split23_out_chain,
  NULL, &g2_5___stem_conv1_stem_Conv_output_0_split23_out_layer, AI_STATIC, 
  .axes = &g2_5___stem_conv1_stem_Conv_output_0_split23_out_axes, 
  .starts = &g2_5___stem_conv1_stem_Conv_output_0_split23_out_starts, 
  .ends = &g2_5___stem_conv1_stem_Conv_output_0_split23_out_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 4, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_layer, 368,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_chain,
  NULL, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output, &g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_layer, 377,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_chain,
  NULL, &_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output, &_choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_0__layers_0_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Concat_output_0_layer, 380,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_0__layers_0_Concat_output_0_chain,
  NULL, &_choice_blocks_0__layers_0_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_0__layers_0_Concat_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_0__layers_0_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_0__layers_0_Transpose_output_0_layer, 386,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_choice_blocks_0__layers_0_Transpose_output_0_chain,
  NULL, &_choice_blocks_0__layers_0_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output, &_choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_0__layers_1_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Concat_output_0_layer, 404,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_0__layers_1_Concat_output_0_chain,
  NULL, &_choice_blocks_0__layers_1_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_0__layers_1_Concat_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_0__layers_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_0__layers_1_Transpose_output_0_layer, 410,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_choice_blocks_0__layers_1_Transpose_output_0_chain,
  NULL, &_choice_blocks_0__layers_1_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output, &_choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_0_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Concat_output_0_layer, 428,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_1__layers_0_Concat_output_0_chain,
  NULL, &_choice_blocks_1__layers_0_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_0_Concat_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_0_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_1__layers_0_Transpose_output_0_layer, 434,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_choice_blocks_1__layers_0_Transpose_output_0_chain,
  NULL, &_choice_blocks_1__layers_0_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output, &_choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_1_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Concat_output_0_layer, 452,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_1__layers_1_Concat_output_0_chain,
  NULL, &_choice_blocks_1__layers_1_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_1_Concat_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_1__layers_1_Transpose_output_0_layer, 458,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_choice_blocks_1__layers_1_Transpose_output_0_chain,
  NULL, &_choice_blocks_1__layers_1_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output, &_choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Concat_output_0_layer, 476,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_1__layers_2_Concat_output_0_chain,
  NULL, &_choice_blocks_1__layers_2_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Concat_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_1__layers_2_Transpose_output_0_layer, 482,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_choice_blocks_1__layers_2_Transpose_output_0_chain,
  NULL, &_choice_blocks_1__layers_2_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_starts_data[] = { 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_ends_data[] = { 4, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_layer, 499,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_starts_data[] = { 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_ends_data[] = { 6, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_layer, 498,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_starts_data[] = { 1, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_ends_data[] = { 8, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_layer, 497,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_starts_data[] = { 3, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_ends_data[] = { 10, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_layer, 496,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_starts_data[] = { 5, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_ends_data[] = { 12, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_layer, 495,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_starts_data[] = { 7, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_ends_data[] = { 14, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_layer, 494,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_starts_data[] = { 9, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_ends_data[] = { 16, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_layer, 493,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_starts_data[] = { 11, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_ends_data[] = { 18, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_layer, 492,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_starts_data[] = { 13, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_ends_data[] = { 20, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_layer, 491,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_starts_data[] = { 15, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_ends_data[] = { 22, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_layer, 490,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 10, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_layer, 606,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_chain,
  NULL, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_starts_data[] = { 17, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_ends_data[] = { 24, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_layer, 489,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_ends, 
)


AI_STATIC_CONST ai_u8 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_axes_data[] = { 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_axes, AI_ARRAY_FORMAT_U8,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_axes_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_starts_data[] = { 19, 0 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_starts, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_starts_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_ends_data[] = { 24, 24 };
AI_ARRAY_OBJ_DECLARE(
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_ends, AI_ARRAY_FORMAT_S16,
    g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_ends_data, g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_1__layers_2_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_layer, 488,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_chain,
  NULL, &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_layer, AI_STATIC, 
  .axes = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_axes, 
  .starts = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_starts, 
  .ends = &g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_layer, 605,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_chain,
  NULL, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output, &g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_layer, 611,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_chain,
  NULL, &_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output, &_choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_0_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Concat_output_0_layer, 614,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_2__layers_0_Concat_output_0_chain,
  NULL, &_choice_blocks_2__layers_0_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_0_Concat_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_0_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_2__layers_0_Transpose_output_0_layer, 620,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_choice_blocks_2__layers_0_Transpose_output_0_chain,
  NULL, &_choice_blocks_2__layers_0_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output, &_choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_1_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Concat_output_0_layer, 638,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_choice_blocks_2__layers_1_Concat_output_0_chain,
  NULL, &_choice_blocks_2__layers_1_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_1_Concat_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _choice_blocks_2__layers_1_Transpose_output_0_layer, 644,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_choice_blocks_2__layers_1_Transpose_output_0_chain,
  NULL, &_choice_blocks_2__layers_1_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _global_pooling_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_choice_blocks_2__layers_1_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_global_pooling_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _global_pooling_GlobalAveragePool_output_0_layer, 650,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_global_pooling_GlobalAveragePool_output_0_chain,
  NULL, &_global_pooling_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(12, 12), 
  .pool_stride = AI_SHAPE_2D_INIT(12, 12), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_global_pooling_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &output_QuantizeLinear_Input_weights, &output_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  output_QuantizeLinear_Input_layer, 656,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &output_QuantizeLinear_Input_chain,
  NULL, &output_QuantizeLinear_Input_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_transpose_input_Transpose(_stai_network_context* net_ctx)
{
  input_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  input_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  input_Transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  input_Transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, { input_output.data->data});
  forward_transpose(&input_Transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, { input_Transpose_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split0_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 7512);
  g2_5___stem_conv1_stem_Conv_output_0_split0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7512);
  _STAI_NETWORK_EVENT_NODE_START_CB(166, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(166, 1, { g2_5___stem_conv1_stem_Conv_output_0_split0_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split1_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split1_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 7984);
  g2_5___stem_conv1_stem_Conv_output_0_split1_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7984);
  _STAI_NETWORK_EVENT_NODE_START_CB(165, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split1_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(165, 1, { g2_5___stem_conv1_stem_Conv_output_0_split1_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split2_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split2_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 8272);
  g2_5___stem_conv1_stem_Conv_output_0_split2_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8272);
  _STAI_NETWORK_EVENT_NODE_START_CB(164, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split2_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(164, 1, { g2_5___stem_conv1_stem_Conv_output_0_split2_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split3_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split3_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 8560);
  g2_5___stem_conv1_stem_Conv_output_0_split3_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8560);
  _STAI_NETWORK_EVENT_NODE_START_CB(163, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split3_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(163, 1, { g2_5___stem_conv1_stem_Conv_output_0_split3_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split4_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split4_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 8848);
  g2_5___stem_conv1_stem_Conv_output_0_split4_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8848);
  _STAI_NETWORK_EVENT_NODE_START_CB(162, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split4_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(162, 1, { g2_5___stem_conv1_stem_Conv_output_0_split4_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split5_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split5_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9136);
  g2_5___stem_conv1_stem_Conv_output_0_split5_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9136);
  _STAI_NETWORK_EVENT_NODE_START_CB(161, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split5_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(161, 1, { g2_5___stem_conv1_stem_Conv_output_0_split5_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split6_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split6_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9424);
  g2_5___stem_conv1_stem_Conv_output_0_split6_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9424);
  _STAI_NETWORK_EVENT_NODE_START_CB(160, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split6_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(160, 1, { g2_5___stem_conv1_stem_Conv_output_0_split6_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split7_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split7_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9712);
  g2_5___stem_conv1_stem_Conv_output_0_split7_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9712);
  _STAI_NETWORK_EVENT_NODE_START_CB(159, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split7_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(159, 1, { g2_5___stem_conv1_stem_Conv_output_0_split7_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split8_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split8_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 10000);
  g2_5___stem_conv1_stem_Conv_output_0_split8_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10000);
  _STAI_NETWORK_EVENT_NODE_START_CB(158, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split8_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(158, 1, { g2_5___stem_conv1_stem_Conv_output_0_split8_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split9_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split9_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 10288);
  g2_5___stem_conv1_stem_Conv_output_0_split9_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10288);
  _STAI_NETWORK_EVENT_NODE_START_CB(157, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split9_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(157, 1, { g2_5___stem_conv1_stem_Conv_output_0_split9_out_output.data->data});
}
void forward_lite_concat_g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out(_stai_network_context* net_ctx)
{
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 7696);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7696);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 7984);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7984);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8272);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8272);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8560);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8560);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8848);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8848);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9136);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9136);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9424);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9424);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9712);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9712);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10000);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10000);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10288);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10288);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 10576);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10576);
  _STAI_NETWORK_EVENT_NODE_START_CB(370, 10, { g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output.data->data});
  forward_concat(&g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(370, 1, { g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split10_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split10_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___stem_conv1_stem_Conv_output_0_split10_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  _STAI_NETWORK_EVENT_NODE_START_CB(156, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split10_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(156, 1, { g2_5___stem_conv1_stem_Conv_output_0_split10_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split11_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split11_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___stem_conv1_stem_Conv_output_0_split11_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  _STAI_NETWORK_EVENT_NODE_START_CB(155, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split11_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(155, 1, { g2_5___stem_conv1_stem_Conv_output_0_split11_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split12_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split12_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___stem_conv1_stem_Conv_output_0_split12_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  _STAI_NETWORK_EVENT_NODE_START_CB(154, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split12_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(154, 1, { g2_5___stem_conv1_stem_Conv_output_0_split12_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split13_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split13_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___stem_conv1_stem_Conv_output_0_split13_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  _STAI_NETWORK_EVENT_NODE_START_CB(153, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split13_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(153, 1, { g2_5___stem_conv1_stem_Conv_output_0_split13_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split14_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split14_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___stem_conv1_stem_Conv_output_0_split14_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  _STAI_NETWORK_EVENT_NODE_START_CB(152, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split14_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(152, 1, { g2_5___stem_conv1_stem_Conv_output_0_split14_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split15_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split15_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___stem_conv1_stem_Conv_output_0_split15_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  _STAI_NETWORK_EVENT_NODE_START_CB(151, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split15_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(151, 1, { g2_5___stem_conv1_stem_Conv_output_0_split15_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split16_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split16_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___stem_conv1_stem_Conv_output_0_split16_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  _STAI_NETWORK_EVENT_NODE_START_CB(150, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split16_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(150, 1, { g2_5___stem_conv1_stem_Conv_output_0_split16_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split17_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split17_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13472);
  g2_5___stem_conv1_stem_Conv_output_0_split17_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13472);
  _STAI_NETWORK_EVENT_NODE_START_CB(149, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split17_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(149, 1, { g2_5___stem_conv1_stem_Conv_output_0_split17_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split18_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split18_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13472);
  g2_5___stem_conv1_stem_Conv_output_0_split18_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13472);
  _STAI_NETWORK_EVENT_NODE_START_CB(148, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split18_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(148, 1, { g2_5___stem_conv1_stem_Conv_output_0_split18_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split19_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split19_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13472);
  g2_5___stem_conv1_stem_Conv_output_0_split19_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13472);
  _STAI_NETWORK_EVENT_NODE_START_CB(147, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split19_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(147, 1, { g2_5___stem_conv1_stem_Conv_output_0_split19_out_output.data->data});
}
void forward_lite_concat_g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out(_stai_network_context* net_ctx)
{
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 7696);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7696);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 7984);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7984);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8272);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8272);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8560);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8560);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8848);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8848);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9136);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9136);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9424);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9424);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 54176);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 54176);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 54464);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 54464);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13456);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13744);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13744);
  _STAI_NETWORK_EVENT_NODE_START_CB(369, 10, { g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_output.data->data});
  forward_concat(&g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(369, 1, { g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split20_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split20_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 54176);
  g2_5___stem_conv1_stem_Conv_output_0_split20_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 54176);
  _STAI_NETWORK_EVENT_NODE_START_CB(146, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split20_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(146, 1, { g2_5___stem_conv1_stem_Conv_output_0_split20_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split21_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split21_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 54176);
  g2_5___stem_conv1_stem_Conv_output_0_split21_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 54176);
  _STAI_NETWORK_EVENT_NODE_START_CB(145, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split21_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(145, 1, { g2_5___stem_conv1_stem_Conv_output_0_split21_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split22_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split22_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 54176);
  g2_5___stem_conv1_stem_Conv_output_0_split22_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 54176);
  _STAI_NETWORK_EVENT_NODE_START_CB(144, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split22_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(144, 1, { g2_5___stem_conv1_stem_Conv_output_0_split22_out_output.data->data});
}
void forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split23_out(_stai_network_context* net_ctx)
{
  _stem_conv1_stem_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17312);
  _stem_conv1_stem_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17312);
  g2_5___stem_conv1_stem_Conv_output_0_split23_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 54176);
  g2_5___stem_conv1_stem_Conv_output_0_split23_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 54176);
  _STAI_NETWORK_EVENT_NODE_START_CB(143, 1, { _stem_conv1_stem_Conv_output_0_output.data->data});
  forward_slice(&g2_5___stem_conv1_stem_Conv_output_0_split23_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(143, 1, { g2_5___stem_conv1_stem_Conv_output_0_split23_out_output.data->data});
}
void forward_lite_concat_g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out(_stai_network_context* net_ctx)
{
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 7696);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7696);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 7984);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7984);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8272);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8272);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8560);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8560);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 8848);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8848);
  _STAI_NETWORK_EVENT_NODE_START_CB(368, 4, { g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_output.data->data});
  forward_concat(&g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(368, 1, { g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output.data->data});
}
void forward_lite_concat__choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0(_stai_network_context* net_ctx)
{
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 10576);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10576);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 13744);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13744);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 8848);
  g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8848);
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 16624);
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16624);
  _STAI_NETWORK_EVENT_NODE_START_CB(377, 3, { g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output.data->data,g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out_output.data->data});
  forward_concat(&_choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(377, 1, { _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output.data->data});
}
void forward_lite_concat__choice_blocks_0__layers_0_Concat_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 16624);
  _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16624);
  _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 600);
  _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 600);
  _choice_blocks_0__layers_0_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 23536);
  _choice_blocks_0__layers_0_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 23536);
  _STAI_NETWORK_EVENT_NODE_START_CB(380, 2, { _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output.data->data,_choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output.data->data});
  forward_concat(&_choice_blocks_0__layers_0_Concat_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(380, 1, { _choice_blocks_0__layers_0_Concat_output_0_output.data->data});
}
void forward_lite_transpose__choice_blocks_0__layers_0_Transpose_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_0__layers_0_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 23536);
  _choice_blocks_0__layers_0_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 23536);
  _choice_blocks_0__layers_0_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_0__layers_0_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _STAI_NETWORK_EVENT_NODE_START_CB(386, 1, { _choice_blocks_0__layers_0_Concat_output_0_output0.data->data});
  forward_transpose(&_choice_blocks_0__layers_0_Transpose_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(386, 1, { _choice_blocks_0__layers_0_Transpose_output_0_output.data->data});
}
void forward_lite_concat__choice_blocks_0__layers_1_Concat_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 14472);
  _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14472);
  _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 21384);
  _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 21384);
  _choice_blocks_0__layers_1_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_0__layers_1_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _STAI_NETWORK_EVENT_NODE_START_CB(404, 2, { _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output.data->data,_choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output.data->data});
  forward_concat(&_choice_blocks_0__layers_1_Concat_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(404, 1, { _choice_blocks_0__layers_1_Concat_output_0_output.data->data});
}
void forward_lite_transpose__choice_blocks_0__layers_1_Transpose_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_0__layers_1_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_0__layers_1_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_0__layers_1_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 14256);
  _choice_blocks_0__layers_1_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14256);
  _STAI_NETWORK_EVENT_NODE_START_CB(410, 1, { _choice_blocks_0__layers_1_Concat_output_0_output0.data->data});
  forward_transpose(&_choice_blocks_0__layers_1_Transpose_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(410, 1, { _choice_blocks_0__layers_1_Transpose_output_0_output.data->data});
}
void forward_lite_concat__choice_blocks_1__layers_0_Concat_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 43616);
  _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 43616);
  _choice_blocks_1__layers_0_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_1__layers_0_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _STAI_NETWORK_EVENT_NODE_START_CB(428, 2, { _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_output.data->data,_choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output.data->data});
  forward_concat(&_choice_blocks_1__layers_0_Concat_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(428, 1, { _choice_blocks_1__layers_0_Concat_output_0_output.data->data});
}
void forward_lite_transpose__choice_blocks_1__layers_0_Transpose_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_0_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_1__layers_0_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_1__layers_0_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_0_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  _STAI_NETWORK_EVENT_NODE_START_CB(434, 1, { _choice_blocks_1__layers_0_Concat_output_0_output0.data->data});
  forward_transpose(&_choice_blocks_1__layers_0_Transpose_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(434, 1, { _choice_blocks_1__layers_0_Transpose_output_0_output.data->data});
}
void forward_lite_concat__choice_blocks_1__layers_1_Concat_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 864);
  _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 864);
  _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 15004);
  _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 15004);
  _choice_blocks_1__layers_1_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28828);
  _choice_blocks_1__layers_1_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28828);
  _STAI_NETWORK_EVENT_NODE_START_CB(452, 2, { _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output.data->data,_choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output.data->data});
  forward_concat(&_choice_blocks_1__layers_1_Concat_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(452, 1, { _choice_blocks_1__layers_1_Concat_output_0_output.data->data});
}
void forward_lite_transpose__choice_blocks_1__layers_1_Transpose_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_1_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28828);
  _choice_blocks_1__layers_1_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28828);
  _choice_blocks_1__layers_1_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_1__layers_1_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _STAI_NETWORK_EVENT_NODE_START_CB(458, 1, { _choice_blocks_1__layers_1_Concat_output_0_output0.data->data});
  forward_transpose(&_choice_blocks_1__layers_1_Transpose_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(458, 1, { _choice_blocks_1__layers_1_Transpose_output_0_output.data->data});
}
void forward_lite_concat__choice_blocks_1__layers_2_Concat_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28512);
  _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28512);
  _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 43616);
  _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 43616);
  _choice_blocks_1__layers_2_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_1__layers_2_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _STAI_NETWORK_EVENT_NODE_START_CB(476, 2, { _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_output.data->data,_choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_output.data->data});
  forward_concat(&_choice_blocks_1__layers_2_Concat_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(476, 1, { _choice_blocks_1__layers_2_Concat_output_0_output.data->data});
}
void forward_lite_transpose__choice_blocks_1__layers_2_Transpose_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_1__layers_2_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 432);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  _STAI_NETWORK_EVENT_NODE_START_CB(482, 1, { _choice_blocks_1__layers_2_Concat_output_0_output0.data->data});
  forward_transpose(&_choice_blocks_1__layers_2_Transpose_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(482, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 8016);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8016);
  _STAI_NETWORK_EVENT_NODE_START_CB(499, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(499, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 8592);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8592);
  _STAI_NETWORK_EVENT_NODE_START_CB(498, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(498, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9168);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9168);
  _STAI_NETWORK_EVENT_NODE_START_CB(497, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(497, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9744);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9744);
  _STAI_NETWORK_EVENT_NODE_START_CB(496, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(496, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 10320);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10320);
  _STAI_NETWORK_EVENT_NODE_START_CB(495, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(495, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 10896);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10896);
  _STAI_NETWORK_EVENT_NODE_START_CB(494, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(494, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 11472);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 11472);
  _STAI_NETWORK_EVENT_NODE_START_CB(493, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(493, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 12048);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12048);
  _STAI_NETWORK_EVENT_NODE_START_CB(492, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(492, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 12048);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12048);
  _STAI_NETWORK_EVENT_NODE_START_CB(491, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(491, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 20016);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 20016);
  _STAI_NETWORK_EVENT_NODE_START_CB(490, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(490, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out_output.data->data});
}
void forward_lite_concat_g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out(_stai_network_context* net_ctx)
{
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8016);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8016);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8592);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8592);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9168);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9168);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9744);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9744);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10320);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10320);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10896);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10896);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 11472);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 11472);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 56832);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 56832);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 57408);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57408);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 57984);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57984);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 22320);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 22320);
  _STAI_NETWORK_EVENT_NODE_START_CB(606, 10, { g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_output.data->data});
  forward_concat(&g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(606, 1, { g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 14256);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14256);
  _STAI_NETWORK_EVENT_NODE_START_CB(489, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(489, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out_output.data->data});
}
void forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out(_stai_network_context* net_ctx)
{
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 28080);
  _choice_blocks_1__layers_2_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 28080);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9264);
  g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9264);
  _STAI_NETWORK_EVENT_NODE_START_CB(488, 1, { _choice_blocks_1__layers_2_Transpose_output_0_output0.data->data});
  forward_slice(&g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(488, 1, { g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out_output.data->data});
}
void forward_lite_concat_g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out(_stai_network_context* net_ctx)
{
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 8688);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8688);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 9264);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9264);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9840);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9840);
  _STAI_NETWORK_EVENT_NODE_START_CB(605, 2, { g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_output.data->data});
  forward_concat(&g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(605, 1, { g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output.data->data});
}
void forward_lite_concat__choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0(_stai_network_context* net_ctx)
{
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 22320);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 22320);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 9840);
  g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9840);
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10992);
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10992);
  _STAI_NETWORK_EVENT_NODE_START_CB(611, 2, { g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out_output.data->data,g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out_output.data->data});
  forward_concat(&_choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(611, 1, { _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output.data->data});
}
void forward_lite_concat__choice_blocks_2__layers_0_Concat_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10992);
  _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10992);
  _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1104);
  _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1104);
  _choice_blocks_2__layers_0_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17904);
  _choice_blocks_2__layers_0_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17904);
  _STAI_NETWORK_EVENT_NODE_START_CB(614, 2, { _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_output.data->data,_choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_output.data->data});
  forward_concat(&_choice_blocks_2__layers_0_Concat_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(614, 1, { _choice_blocks_2__layers_0_Concat_output_0_output.data->data});
}
void forward_lite_transpose__choice_blocks_2__layers_0_Transpose_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_2__layers_0_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 17904);
  _choice_blocks_2__layers_0_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 17904);
  _choice_blocks_2__layers_0_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _choice_blocks_2__layers_0_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(620, 1, { _choice_blocks_2__layers_0_Concat_output_0_output0.data->data});
  forward_transpose(&_choice_blocks_2__layers_0_Transpose_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(620, 1, { _choice_blocks_2__layers_0_Transpose_output_0_output.data->data});
}
void forward_lite_concat__choice_blocks_2__layers_1_Concat_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 14688);
  _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14688);
  _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 21600);
  _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 21600);
  _choice_blocks_2__layers_1_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _choice_blocks_2__layers_1_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(638, 2, { _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_output.data->data,_choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_output.data->data});
  forward_concat(&_choice_blocks_2__layers_1_Concat_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(638, 1, { _choice_blocks_2__layers_1_Concat_output_0_output.data->data});
}
void forward_lite_transpose__choice_blocks_2__layers_1_Transpose_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_2__layers_1_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _choice_blocks_2__layers_1_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _choice_blocks_2__layers_1_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 13824);
  _choice_blocks_2__layers_1_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13824);
  _STAI_NETWORK_EVENT_NODE_START_CB(644, 1, { _choice_blocks_2__layers_1_Concat_output_0_output0.data->data});
  forward_transpose(&_choice_blocks_2__layers_1_Transpose_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(644, 1, { _choice_blocks_2__layers_1_Transpose_output_0_output.data->data});
}
void forward_lite_ap_integer_INT8__global_pooling_GlobalAveragePool_output_0(_stai_network_context* net_ctx)
{
  _choice_blocks_2__layers_1_Transpose_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 13824);
  _choice_blocks_2__layers_1_Transpose_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13824);
  _global_pooling_GlobalAveragePool_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _global_pooling_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(650, 1, { _choice_blocks_2__layers_1_Transpose_output_0_output0.data->data});
  forward_ap_integer_INT8(&_global_pooling_GlobalAveragePool_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(650, 1, { _global_pooling_GlobalAveragePool_output_0_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_output_QuantizeLinear_Input(_stai_network_context* net_ctx)
{
  _global_pooling_GlobalAveragePool_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _global_pooling_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  output_QuantizeLinear_Input_weights_array.data = AI_PTR(net_ctx->_weights[0] + 43232);
  output_QuantizeLinear_Input_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 43232);
  output_QuantizeLinear_Input_bias_array.data = AI_PTR(net_ctx->_weights[0] + 52832);
  output_QuantizeLinear_Input_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 52832);
  output_QuantizeLinear_Input_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 96);
  output_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 96);
  output_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  output_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(656, 1, { _global_pooling_GlobalAveragePool_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&output_QuantizeLinear_Input_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(656, 1, { output_QuantizeLinear_Input_output.data->data});
}

/*****************************************************************************/



static const ai_u16 _stem_conv0_stem_Conv_output_0_t_in_0_shape_w_const_u16 = 96;
static const ai_u16 _stem_conv0_stem_Conv_output_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_u16 _stem_conv0_stem_Conv_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_i32 _stem_conv0_stem_Conv_output_0_l_pad_W_0_const_s32 = 1;
static const ai_u16 _stem_conv0_stem_Conv_output_0_l_stride_0_const_u16 = 2;
static const ai_i8 _stem_conv0_stem_Conv_output_0_t_in_0_fmt_zero_const_s8 = 0;
static const ai_i8 _stem_conv0_stem_Conv_output_0_t_out_0_fmt_zero_const_s8 = 4;
static const ai_float _stem_conv0_stem_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.007843106053769588f;
static const ai_float _stem_conv0_stem_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.039968304336071014f;
static const ai_float _stem_conv0_stem_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00670353090390563f, 0.006873375736176968f, 0.00466562807559967f, 0.005345245823264122f, 0.004788680467754602f, 0.005202106665819883f, 0.005770521704107523f, 0.00734195439144969f, 0.006471225526183844f, 0.005404340103268623f, 0.009881202131509781f, 0.004572231788188219f, 0.005565650761127472f, 0.005345826502889395f, 0.005213427357375622f, 0.005761161912232637f);
static const ai_layer_format_type _stem_conv0_stem_Conv_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _stem_conv0_stem_Conv_output_0_t_out_0_shape_w_const_u16 = 48;

static const ai_u16 _stem_conv1_stem_Conv_output_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 _stem_conv1_stem_Conv_output_0_t_in_0_shape_h_const_u16 = 48;
static const ai_u16 _stem_conv1_stem_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _stem_conv1_stem_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _stem_conv1_stem_Conv_output_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 _stem_conv1_stem_Conv_output_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 _stem_conv1_stem_Conv_output_0_t_in_0_fmt_zero_const_s8 = 4;
static const ai_i8 _stem_conv1_stem_Conv_output_0_t_out_0_fmt_zero_const_s8 = -5;
static const ai_float _stem_conv1_stem_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.039968304336071014f;
static const ai_float _stem_conv1_stem_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float _stem_conv1_stem_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001151944394223392f, 0.0011858241632580757f, 0.0007211568881757557f, 0.001033114269375801f, 0.0014890078455209732f, 0.0009391389903612435f, 0.001181922503747046f, 0.000919046753551811f, 0.0010968046262860298f, 0.000872393895406276f, 0.0005811158334836364f, 0.0008988796034827828f, 0.0013361976016312838f, 0.001119287684559822f, 0.0014403370441868901f, 0.0008231089450418949f);
static const ai_layer_format_type _stem_conv1_stem_Conv_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 48;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_w_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_h_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16 = 2;
static const ai_i32 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_pad_W_0_const_s32 = 0;
static const ai_i32 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_pad_H_0_const_s32 = 0;
static const ai_i8 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.0012272186577320099f;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0006361189298331738f, 0.0012845080345869064f, 0.000791929429396987f, 0.0013557375641539693f, 0.0008968824404291809f, 0.001379787689074874f, 0.0008874566410668194f, 0.0007736891275271773f, 0.0012636375613510609f, 0.0011256628204137087f, 0.0009520613821223378f, 0.0008407061104662716f);
static const ai_layer_format_type _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_h_const_u16 = 24;

static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16 = 12;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32 = 2;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i8 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8 = 26;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.0012272186577320099f;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.008660058490931988f;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.020473822951316833f, 0.017948582768440247f, 0.016784099861979485f, 0.021232932806015015f, 0.017775462940335274f, 0.016813794150948524f, 0.02341325394809246f, 0.020528927445411682f, 0.019942086189985275f, 0.019948521628975868f, 0.020833348855376244f, 0.024168485775589943f);
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16 = 24;

static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16 = 12;
static const ai_u16 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8 = 26;
static const ai_i8 _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.008660058490931988f;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.001367607619613409f;
static const ai_float _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001483640051446855f, 0.0015914395917207003f, 0.001407262054271996f, 0.0013937643961980939f, 0.0009191372082568705f, 0.0007165040005929768f, 0.0009388168691657484f, 0.001391148311085999f, 0.0018698661588132381f, 0.001216719625517726f, 0.0010951643344014883f, 0.0011635218979790807f);
static const ai_layer_format_type _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16 = 3;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_H_0_const_s32 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32 = 0.018758021295070648f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32 = 0.018758021295070648f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32 = 0.0026632575318217278f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32 = 0.024940621107816696f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32 = 0.024940621107816696f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32 = 0.004274711012840271f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32 = 0.023928988724946976f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32 = 0.023928988724946976f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32 = 0.004869819153100252f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32 = 0.025861816480755806f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32 = 0.025861816480755806f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32 = 0.006499545648694038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32 = 0.029146652668714523f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32 = 0.029146652668714523f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32 = 0.004104751627892256f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32 = 0.02600790746510029f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32 = 0.02600790746510029f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32 = 0.0047948118299245834f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32 = 0.024355359375476837f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32 = 0.024355359375476837f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32 = 0.0051765721291303635f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32 = 0.023207856342196465f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32 = 0.023207856342196465f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32 = 0.004307139664888382f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32 = 0.02336117997765541f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32 = 0.02336117997765541f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32 = 0.004592200741171837f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32 = 0.03350953757762909f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32 = 0.03350953757762909f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32 = 0.005467839539051056f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32 = 0.02662457898259163f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32 = 0.02662457898259163f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32 = 0.004460128955543041f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32 = 0.025503236800432205f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32 = 0.025503236800432205f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32 = 0.003856317140161991f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_fmt_scale_const_f32 = 0.021442905068397522f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_fmt_scale_const_f32 = 0.021442905068397522f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_fmt_scale_const_f32 = 0.004413940478116274f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_fmt_scale_const_f32 = 0.023899368941783905f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_fmt_scale_const_f32 = 0.023899368941783905f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_fmt_scale_const_f32 = 0.00526389479637146f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_fmt_scale_const_f32 = 0.026593144983053207f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_fmt_scale_const_f32 = 0.026593144983053207f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_fmt_scale_const_f32 = 0.005249801557511091f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_fmt_scale_const_f32 = 0.026758186519145966f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_fmt_scale_const_f32 = 0.026758186519145966f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_fmt_scale_const_f32 = 0.0037528062239289284f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_fmt_scale_const_f32 = 0.029495229944586754f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_fmt_scale_const_f32 = 0.029495229944586754f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_fmt_scale_const_f32 = 0.004686704371124506f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_fmt_scale_const_f32 = 0.021852176636457443f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_fmt_scale_const_f32 = 0.021852176636457443f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_fmt_scale_const_f32 = 0.004232417792081833f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_fmt_scale_const_f32 = 0.028399022296071053f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_fmt_scale_const_f32 = 0.028399022296071053f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_fmt_scale_const_f32 = 0.004691829439252615f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_fmt_scale_const_f32 = 0.03069116733968258f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_fmt_scale_const_f32 = 0.03069116733968258f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_fmt_scale_const_f32 = 0.005343812983483076f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_fmt_scale_const_f32 = 0.022025588899850845f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_fmt_scale_const_f32 = 0.022025588899850845f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_fmt_scale_const_f32 = 0.005204447079449892f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_fmt_scale_const_f32 = 0.024105586111545563f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_fmt_scale_const_f32 = 0.024105586111545563f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_fmt_scale_const_f32 = 0.004810446407645941f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_fmt_scale_const_f32 = 0.026857009157538414f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_fmt_scale_const_f32 = 0.026857009157538414f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_fmt_scale_const_f32 = 0.0047269705682992935f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_shape_w_const_u16 = 48;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_stride_1_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_stride_0_const_u16 = 1;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_fmt_scale_const_f32 = 0.01088507566601038f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_fmt_scale_const_f32 = 0.0266192015260458f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0182851180434227f, 0.023584868758916855f, 0.02240690030157566f, 0.013524525798857212f, 0.012701394036412239f, 0.020248930901288986f, 0.018524600192904472f, 0.018023673444986343f, 0.0253917146474123f, 0.013381202705204487f, 0.01807357184588909f, 0.015350822359323502f, 0.024489134550094604f, 0.023960484191775322f, 0.01730942539870739f, 0.02088613621890545f);
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_shape_w_const_u16 = 47;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_l_stride_1_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_l_stride_0_const_u16 = 2;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_fmt_scale_const_f32 = 0.0266192015260458f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_fmt_scale_const_f32 = 0.004641928244382143f;
static const ai_float g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007016301969997585f, 0.0008311352576129138f, 0.001004962483420968f, 0.0012563184136524796f, 0.0007701343856751919f, 0.0007866955711506307f, 0.0014423850225284696f, 0.0010695771779865026f, 0.0011070301989093423f, 0.0009507936774753034f, 0.0012622056528925896f, 0.0008711042464710772f);
static const ai_layer_format_type g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;





static const ai_u16 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.006499545648694038f;
static const ai_float _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.00047203159192577004f;
static const ai_float _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0008025523857213557f, 0.0008527628378942609f, 0.0006432285299524665f, 0.0010187760926783085f, 0.000804055598564446f, 0.0007268293993547559f, 0.0008865093113854527f, 0.0006113631534390152f, 0.0009644049569033086f, 0.0006915805279277265f, 0.0008377752965316176f, 0.0007635481306351721f);
static const ai_layer_format_type _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.006499545648694038f;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.0008584044990129769f;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0006587420357391238f, 0.0009830592898651958f, 0.0007623783894814551f, 0.0005751181161031127f, 0.0004997225478291512f, 0.0007806340581737459f, 0.000542315247002989f, 0.0007172526093199849f, 0.0007739209686405957f, 0.0014346186071634293f, 0.0010384945198893547f, 0.0007604953134432435f);
static const ai_layer_format_type _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16 = 12;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16 = 5;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16 = 5;
static const ai_i32 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32 = 2;
static const ai_i32 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32 = 2;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i8 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8 = 24;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.0008584044990129769f;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.003474349156022072f;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.009842833504080772f, 0.02138834446668625f, 0.013141813687980175f, 0.01124612707644701f, 0.02088376134634018f, 0.01271058339625597f, 0.014704262837767601f, 0.01439753919839859f, 0.017235025763511658f, 0.013429347425699234f, 0.011656045913696289f, 0.011403205804526806f);
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16 = 24;

static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16 = 12;
static const ai_u16 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8 = 24;
static const ai_i8 _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.003474349156022072f;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.0005766698159277439f;
static const ai_float _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0014256448484957218f, 0.0019549252465367317f, 0.0020317109301686287f, 0.0012193868169561028f, 0.001704351045191288f, 0.0020023323595523834f, 0.0013195301871746778f, 0.0010311870137229562f, 0.001214688061736524f, 0.0011773970909416676f, 0.0014252124819904566f, 0.001272906200028956f);
static const ai_layer_format_type _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.0005766698159277439f;
static const ai_float _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 6.87527353875339e-05f;
static const ai_float _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007722160080447793f, 0.0006617612089030445f, 0.0013590065063908696f, 0.0008493677596561611f, 0.0011417532805353403f, 0.0007102708332240582f, 0.0005692360573448241f, 0.0006145501974970102f, 0.0008498575771227479f, 0.0010399841703474522f, 0.0007058788905851543f, 0.0008269600220955908f, 0.0007572287577204406f, 0.0006123311468400061f, 0.000819830340333283f, 0.0007016967283561826f, 0.000905984896235168f, 0.0006778671522624791f, 0.0008541778079234064f, 0.0009826663881540298f, 0.0007316875271499157f, 0.0007906476967036724f, 0.0009544404456391931f, 0.0008570820791646838f);
static const ai_layer_format_type _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.0005766698159277439f;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 8.472690387861803e-05f;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0008524278528057039f, 0.0007444784278050065f, 0.0007644871948286891f, 0.0006216891924850643f, 0.0008485013386234641f, 0.0008001690148375928f, 0.0007524506654590368f, 0.0008484015124849975f, 0.0008600233704783022f, 0.0007495400495827198f, 0.000658029573969543f, 0.0005791661096736789f, 0.0008043615962378681f, 0.0006381366401910782f, 0.0010690095368772745f, 0.0007122547831386328f, 0.0010556841734796762f, 0.0007678267429582775f, 0.0009300410165451467f, 0.0007065330282784998f, 0.0008661466999910772f, 0.0007932137814350426f, 0.0005787730915471911f, 0.0008239806047640741f);
static const ai_layer_format_type _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_shape_h_const_u32 = 24;

static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16 = 26;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16 = 26;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i8 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8 = 33;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32 = 8.472690387861803e-05f;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.001381843350827694f;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.010568423196673393f, 0.021067069843411446f, 0.008228532038629055f, 0.015625666826963425f, 0.016528749838471413f, 0.01387418806552887f, 0.01782066375017166f, 0.014168618246912956f, 0.006667685694992542f, 0.013274972327053547f, 0.009077556431293488f, 0.006511348765343428f, 0.012852063402533531f, 0.013710695318877697f, 0.008073496632277966f, 0.014917452819645405f, 0.019463034346699715f, 0.01210737507790327f, 0.01266355998814106f, 0.020000673830509186f, 0.01115579716861248f, 0.009603830985724926f, 0.006807906553149223f, 0.010396236553788185f);
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16 = 24;

static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8 = 33;
static const ai_i8 _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.001381843350827694f;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.0001844168291427195f;
static const ai_float _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0006640080828219652f, 0.0005707506788894534f, 0.0008444999111816287f, 0.0006694328039884567f, 0.0008856671047396958f, 0.0007433638093061745f, 0.0007707644836045802f, 0.0007771790260449052f, 0.0009853756055235863f, 0.0006776622612960637f, 0.0007358234724961221f, 0.0005236914730630815f, 0.0010629614116623998f, 0.0004532260063569993f, 0.0006792050553485751f, 0.0005928758764639497f, 0.0010389311937615275f, 0.0006386530585587025f, 0.0009479423752054572f, 0.0007305754697881639f, 0.0006972157279960811f, 0.0009898408316075802f, 0.0009365917067043483f, 0.0011479698587208986f);
static const ai_layer_format_type _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.0001844168291427195f;
static const ai_float _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 4.599483509082347e-05f;
static const ai_float _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0005103362491354346f, 0.00044167941086925566f, 0.0003472172247711569f, 0.00031489456887356937f, 0.0003169910341966897f, 0.0004721819132100791f, 0.00044051610166206956f, 0.00046236556954681873f, 0.00041011383291333914f, 0.0004602911358233541f, 0.0003978725289925933f, 0.0004998037475161254f, 0.00047094482579268515f, 0.0004025496600661427f, 0.000493433850351721f, 0.00045879415119998157f, 0.00038475304609164596f, 0.00042663089698180556f, 0.0004438639443833381f, 0.0004746335034724325f, 0.00039259251207113266f, 0.0005148036871105433f, 0.0004292354569770396f, 0.0004271136422175914f);
static const ai_layer_format_type _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.0001844168291427195f;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 3.1744391890242696e-05f;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0005529298796318471f, 0.00030919769778847694f, 0.0003775278164539486f, 0.00034904785570688546f, 0.0003472406533546746f, 0.00039047221071086824f, 0.000522277841810137f, 0.00034065646468661726f, 0.0006127672386355698f, 0.00045199645683169365f, 0.00045166834024712443f, 0.00044871363206766546f, 0.0003785754961427301f, 0.0003427852934692055f, 0.0005028418963775039f, 0.0005499366670846939f, 0.0004785486089531332f, 0.0005836377386003733f, 0.0004935336764901876f, 0.00044527873978950083f, 0.0006899716681800783f, 0.00048406177666038275f, 0.0002951745118480176f, 0.0005881763063371181f);
static const ai_layer_format_type _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_shape_h_const_u32 = 24;

static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16 = 26;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16 = 26;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i8 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8 = 19;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32 = 3.1744391890242696e-05f;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.0012510142987594008f;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.016285263001918793f, 0.016658827662467957f, 0.015389367938041687f, 0.009776771999895573f, 0.008827481418848038f, 0.011381804011762142f, 0.014202984049916267f, 0.0099330460652709f, 0.011645065620541573f, 0.013562406413257122f, 0.007611995097249746f, 0.013007826171815395f, 0.011769494041800499f, 0.011612262576818466f, 0.015354905277490616f, 0.015165752731263638f, 0.013312170282006264f, 0.01089995913207531f, 0.013805360533297062f, 0.013096099719405174f, 0.010848292149603367f, 0.008650921285152435f, 0.015089713037014008f, 0.014079861342906952f);
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16 = 24;

static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8 = 19;
static const ai_i8 _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.0012510142987594008f;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.00013865303480997682f;
static const ai_float _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0008492927299812436f, 0.000767749734222889f, 0.0009073499822989106f, 0.0007880802149884403f, 0.0010213529458269477f, 0.0006314528873190284f, 0.000857626786455512f, 0.0007084751850925386f, 0.0006695009651593864f, 0.0005857534124515951f, 0.0009435360552743077f, 0.0008733018767088652f, 0.000652025977615267f, 0.0009538538288325071f, 0.0008919090032577515f, 0.0005147496704012156f, 0.0007173100020736456f, 0.000691770575940609f, 0.0008537929970771074f, 0.0006494437693618238f, 0.0008400384103879333f, 0.0010641474509611726f, 0.0006588833639398217f, 0.0007681791321374476f);
static const ai_layer_format_type _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.00013865303480997682f;
static const ai_float _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 4.370134411146864e-05f;
static const ai_float _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0003015519178006798f, 0.0003565273364074528f, 0.0003521075122989714f, 0.0003912092943210155f, 0.0004596473590936512f, 0.00040944706415757537f, 0.00035388217656873167f, 0.0003461398009676486f, 0.00036042704596184194f, 0.00045126370969228446f, 0.00038305643829517066f, 0.000466382218291983f, 0.00036803403054364026f, 0.00044502713717520237f, 0.0004148106090724468f, 0.0003448073985055089f, 0.000480589980725199f, 0.0005346221732906997f, 0.00043993006693199277f, 0.0004908928531222045f, 0.0004352763353381306f, 0.00041666682227514684f, 0.00046287369332276285f, 0.0004534399777185172f);
static const ai_layer_format_type _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.00013865303480997682f;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 3.607115286285989e-05f;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0004537711211014539f, 0.0004013098368886858f, 0.0005669159581884742f, 0.00030483692535199225f, 0.00048058590618893504f, 0.00048486687592230737f, 0.00034837168641388416f, 0.0005672266124747694f, 0.0005756212049163878f, 0.00039026193553581834f, 0.0003330805920995772f, 0.0004396400472614914f, 0.0003911562671419233f, 0.00034086761297658086f, 0.000527142605278641f, 0.000343979278113693f, 0.0003741367254406214f, 0.0006260114023461938f, 0.00040807254845276475f, 0.0004596147919073701f, 0.00035049888538196683f, 0.00033330012229271233f, 0.0004645415465347469f, 0.00047679239651188254f);
static const ai_layer_format_type _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_shape_h_const_u32 = 24;

static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16 = 26;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16 = 26;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i8 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8 = 3;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32 = 3.607115286285989e-05f;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.0009328379528596997f;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.011232579126954079f, 0.017525073140859604f, 0.01362321525812149f, 0.015132642351090908f, 0.008207752369344234f, 0.017899151891469955f, 0.016873177140951157f, 0.01219874992966652f, 0.014710638672113419f, 0.008786605671048164f, 0.00989482644945383f, 0.008878245018422604f, 0.025873802602291107f, 0.01499843504279852f, 0.014348120428621769f, 0.014393399469554424f, 0.0133234653621912f, 0.008907242678105831f, 0.010805233381688595f, 0.00872010737657547f, 0.017866304144263268f, 0.008967493660748005f, 0.011974962428212166f, 0.010801460593938828f);
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16 = 24;

static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8 = 3;
static const ai_i8 _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.0009328379528596997f;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007229489856399596f, 0.0007483140798285604f, 0.0006793326465412974f, 0.000560698623303324f, 0.0006753969355486333f, 0.0006400193669833243f, 0.0008037359802983701f, 0.0007478698971681297f, 0.0005295935552567244f, 0.0008896095096133649f, 0.0006406104075722396f, 0.0007537142373621464f, 0.0009856072720140219f, 0.0005674536223523319f, 0.000957160780671984f, 0.0006942698964849114f, 0.0007562154205515981f, 0.0008510719635523856f, 0.0006334120407700539f, 0.0006517363362945616f, 0.0010088986018672585f, 0.0006055226549506187f, 0.0007929668063297868f, 0.0006752921035513282f);
static const ai_layer_format_type _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 24;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_w_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_h_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16 = 2;
static const ai_i8 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 4.013987927464768e-05f;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00044007296673953533f, 0.00037824679748155177f, 0.00034574070014059544f, 0.0003105783252976835f, 0.0004784465709235519f, 0.0003904607438016683f, 0.0003874539688695222f, 0.0003224504180252552f, 0.00033033970976248384f, 0.00032851260039024055f, 0.0005257579032331705f, 0.0004953437601216137f, 0.000573340046685189f, 0.00046789689804427326f, 0.0003619043854996562f, 0.0005115390522405505f, 0.00033013400388881564f, 0.0002876537328120321f, 0.0005289363325573504f, 0.0003892788081429899f, 0.000518356915563345f, 0.000474366097478196f, 0.00047401105985045433f, 0.000553491001483053f, 0.00040084755164571106f, 0.00039906069287098944f, 0.0005873088957741857f, 0.0004568026342894882f, 0.00036371758324094117f, 0.0004437070165295154f, 0.0004715495742857456f, 0.00045474260696209967f, 0.00036858805106021464f, 0.00045916426461189985f, 0.00032426329562440515f, 0.0004260607238393277f, 0.000589529168792069f, 0.0006114974385127425f, 0.00043148480472154915f, 0.00036752282176166773f, 0.0003596889437176287f, 0.0006095819408074021f, 0.000414828973589465f, 0.0005561064463108778f, 0.0004823514318559319f, 0.0003551030531525612f, 0.00041567219886928797f, 0.0005293901194818318f);
static const ai_layer_format_type _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_h_const_u16 = 12;

static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32 = 3;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i8 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8 = -9;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32 = 4.013987927464768e-05f;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.0023874829057604074f;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014159074984490871f, 0.01384306512773037f, 0.014531704597175121f, 0.01313643530011177f, 0.01307313609868288f, 0.01559618953615427f, 0.011173227801918983f, 0.01353683602064848f, 0.013295657932758331f, 0.018159637227654457f, 0.014385226182639599f, 0.01381013449281454f, 0.012101189233362675f, 0.0123746944591403f, 0.015343258157372475f, 0.012098639272153378f, 0.014358951710164547f, 0.01442011073231697f, 0.011925885453820229f, 0.01443345658481121f, 0.011173286475241184f, 0.019777240231633186f, 0.013141114264726639f, 0.012069608084857464f, 0.012638920918107033f, 0.01624348945915699f, 0.015570118092000484f, 0.016891753301024437f, 0.012821957468986511f, 0.01113848201930523f, 0.016049951314926147f, 0.012353939935564995f, 0.012913771905004978f, 0.012080817483365536f, 0.014136281795799732f, 0.01268375851213932f, 0.013179759494960308f, 0.023504221811890602f, 0.014024279080331326f, 0.013822313398122787f, 0.011861048638820648f, 0.011775570921599865f, 0.012345459312200546f, 0.01271753665059805f, 0.01732214353978634f, 0.01551023405045271f, 0.010883420705795288f, 0.01910908706486225f);
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16 = 12;

static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8 = -9;
static const ai_i8 _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.0023874829057604074f;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.00019421213073655963f;
static const ai_float _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0004286783514544368f, 0.00036634309799410403f, 0.00043204636313021183f, 0.000397982686990872f, 0.00037896246067248285f, 0.0004601473337970674f, 0.0003640154027380049f, 0.0005125034367665648f, 0.0004811176913790405f, 0.0003499656158965081f, 0.00039961523725651205f, 0.00037321902345865965f, 0.0004757009446620941f, 0.0005178183782845736f, 0.000710026768501848f, 0.00044006164534948766f, 0.000374280585674569f, 0.0004216392699163407f, 0.000336794852046296f, 0.0003164147783536464f, 0.0005766655085608363f, 0.000408108695410192f, 0.00032261715386994183f, 0.00036517332773655653f, 0.0005252630799077451f, 0.00039799470687285066f, 0.0003868621133733541f, 0.0006097569712437689f, 0.0004092142335139215f, 0.0003631186264101416f, 0.0003381401766091585f, 0.000444665813120082f, 0.00037207832792773843f, 0.0004601757100317627f, 0.00043045348138548434f, 0.0004865928494837135f, 0.0006091144168749452f, 0.0005173863610252738f, 0.00046686214045621455f, 0.00039737430051900446f, 0.00046040353481657803f, 0.0003313413180876523f, 0.0005927063175477087f, 0.0003938862355425954f, 0.0004918547347187996f, 0.0003308943414594978f, 0.0004663348663598299f, 0.0005294043221510947f);
static const ai_layer_format_type _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_H_0_const_s32 = 3;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32 = 0.000938768032938242f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32 = 0.000938768032938242f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32 = 0.00010105739784194157f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_H_0_const_s32 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32 = 0.0009222851949743927f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32 = 0.0009222851949743927f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32 = 9.631847933633253e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32 = 0.0009271063026972115f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32 = 0.0009271063026972115f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32 = 9.459926513954997e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32 = 0.0009271719609387219f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32 = 0.0009271719609387219f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32 = 9.460085129830986e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32 = 0.0009270430309697986f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32 = 0.0009270430309697986f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32 = 9.458923886995763e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32 = 0.0009272801107726991f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32 = 0.0009272801107726991f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32 = 9.461980516789481e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32 = 0.0009270129376091063f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32 = 0.0009270129376091063f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32 = 9.459777356823906e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32 = 0.0009270954178646207f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32 = 0.0009270954178646207f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32 = 9.459783177589998e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32 = 0.0009271095623262227f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32 = 0.0009271095623262227f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32 = 9.461582521907985e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32 = 0.0009271888993680477f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32 = 0.0009271888993680477f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32 = 9.461899753659964e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32 = 0.000927017885260284f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32 = 0.000927017885260284f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32 = 9.404182492289692e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16 = 24;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16 = 5;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_1_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_0_const_u16 = 1;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32 = 0.00012253265595063567f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32 = 0.0009037021081894636f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014834247529506683f, 0.01088247261941433f, 0.012256153859198093f, 0.014073473401367664f, 0.017474336549639702f, 0.012247862294316292f, 0.011323676444590092f, 0.014031989499926567f, 0.013886837288737297f, 0.012332170270383358f, 0.01951136440038681f, 0.011604180559515953f, 0.010896610096096992f, 0.013820890337228775f, 0.02184915542602539f, 0.009678450413048267f, 0.016688095405697823f, 0.01630108617246151f, 0.015309080481529236f, 0.01420516986399889f, 0.012282811105251312f, 0.016154030337929726f, 0.009026291780173779f, 0.01121823862195015f, 0.0135884340852499f, 0.015632309019565582f, 0.01316656731069088f, 0.014755508862435818f, 0.015621316619217396f, 0.01193795446306467f, 0.012510172091424465f, 0.011012003757059574f, 0.02163480781018734f, 0.016360891982913017f, 0.012723350897431374f, 0.014553161337971687f, 0.015773508697748184f, 0.012944437563419342f, 0.01366655994206667f, 0.011781426146626472f, 0.014549868181347847f, 0.011499986983835697f, 0.011657795868813992f, 0.009696324355900288f, 0.01698414236307144f, 0.015169934369623661f, 0.014979793690145016f, 0.012873451225459576f);
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_h_const_u16 = 1;

static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16 = 23;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_1_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_0_const_u16 = 2;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32 = 0.0009037021081894636f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32 = 8.717770833754912e-05f;
static const ai_float g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041359858005307615f, 0.00037066329969093204f, 0.00038788237725384533f, 0.0006189525010995567f, 0.00041348786908201873f, 0.00035467767156660557f, 0.0003436314582359046f, 0.0005338701303116977f, 0.0003542565100360662f, 0.0004621314874384552f, 0.0003322088159620762f, 0.00032026335247792304f, 0.0005682812188751996f, 0.00037645784323103726f, 0.00031346449395641685f, 0.00041316120768897235f, 0.0004232106148265302f, 0.00034559908090159297f, 0.0004194559878669679f, 0.0004453327273949981f, 0.0003842011501546949f, 0.0005041849799454212f, 0.0003197279293090105f, 0.0004390404501464218f, 0.00040483957855030894f, 0.00035098203807137907f, 0.0004932076553814113f, 0.0003356916131451726f, 0.0003521017206367105f, 0.0004976413329131901f, 0.0003084056661464274f, 0.0004055003519169986f, 0.00037767147296108305f, 0.00042071001371368766f, 0.0004227258323226124f, 0.00036552330129779875f, 0.0003541732730809599f, 0.00042879852117039263f, 0.00038060269434936345f, 0.00043090159306302667f, 0.0004860587068833411f, 0.0003743433626368642f, 0.0004503853851929307f, 0.00038568975287489593f, 0.0003819632693193853f, 0.0003735810169018805f, 0.00034597344347275794f, 0.000483574258396402f);
static const ai_layer_format_type g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;





static const ai_u16 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.00019421213073655963f;
static const ai_float _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 4.426796658663079e-05f;
static const ai_float _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00025262642884626985f, 0.00026037616771645844f, 0.00021621500491164625f, 0.00021008455951232463f, 0.00023681242601014674f, 0.0001952331222128123f, 0.00022540608188137412f, 0.0002472777268849313f, 0.00023665318440180272f, 0.00023842512746341527f, 0.00029962489497847855f, 0.00019840558525174856f, 0.0002329387643840164f, 0.0002461086551193148f, 0.0002030834584729746f, 0.00023690698435530066f, 0.0002679295721463859f, 0.00020839284115936607f, 0.00020083568233530968f, 0.00026507972506806254f, 0.000211838967516087f, 0.00024445392773486674f, 0.00020415733160916716f, 0.0001813089766073972f, 0.0002892495831474662f, 0.00028003062470816076f, 0.0002748348342720419f, 0.00022877209994476289f, 0.0002061378036160022f, 0.0002803436655085534f, 0.00024098108406178653f, 0.00025507627287879586f, 0.00025080793420784175f, 0.00019852365949191153f, 0.00022017267474438995f, 0.00020545897132251412f, 0.00025555407046340406f, 0.0002223056653747335f, 0.00021902176376897842f, 0.00021579851454589516f, 0.0003068678197450936f, 0.00026125137810595334f, 0.00023289916862268f, 0.00018350592290516943f, 0.00024880835553631186f, 0.00030495223472826183f, 0.00024868902983143926f, 0.00017988534818869084f);
static const ai_layer_format_type _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.00019421213073655963f;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32 = 4.1332536056870595e-05f;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0002823158574756235f, 0.00020626973127946258f, 0.0002426937426207587f, 0.0003406598116271198f, 0.00026356056332588196f, 0.0002415852213744074f, 0.000346023531164974f, 0.00025294808438047767f, 0.00024882296565920115f, 0.00020928788580931723f, 0.00022781570442020893f, 0.0001805905922083184f, 0.00026112477644346654f, 0.0002046897279797122f, 0.00019457076268736273f, 0.0002998346171807498f, 0.0002601113519631326f, 0.00031678666709922254f, 0.00032963810372166336f, 0.0002168897190131247f, 0.00029328951495699584f, 0.00015769996389281005f, 0.0002584956237114966f, 0.00021260695939417928f, 0.0002546119794715196f, 0.00030321741360239685f, 0.0002230110694654286f, 0.00022797673591412604f, 0.0002559380081947893f, 0.0002371809969190508f, 0.0002058063546428457f, 0.00021899690909776837f, 0.0002698423049878329f, 0.00019583379616960883f, 0.0003248521825298667f, 0.0002249544922960922f, 0.00023135477385949343f, 0.0002966380270663649f, 0.00022999422799330205f, 0.00022832330432720482f, 0.00022441506735049188f, 0.0003253419999964535f, 0.0002126305626006797f, 0.00022472137061413378f, 0.00021390296751633286f, 0.00017747387755662203f, 0.000265582901192829f, 0.00020849803695455194f);
static const ai_layer_format_type _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16 = 7;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16 = 7;
static const ai_i32 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32 = 3;
static const ai_i32 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32 = 3;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i8 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8 = -19;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32 = 4.1332536056870595e-05f;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.001747139380313456f;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.015094181522727013f, 0.012185691855847836f, 0.015162237919867039f, 0.014920187182724476f, 0.01401170901954174f, 0.01395522803068161f, 0.013378545641899109f, 0.010304891504347324f, 0.012595889158546925f, 0.01599819026887417f, 0.018501725047826767f, 0.015488320030272007f, 0.012026670388877392f, 0.010032129473984241f, 0.01289375964552164f, 0.014821605756878853f, 0.014513724483549595f, 0.012205093167722225f, 0.013243135064840317f, 0.011096376925706863f, 0.015177393332123756f, 0.01734154298901558f, 0.016768569126725197f, 0.010360252112150192f, 0.012294186279177666f, 0.01249554194509983f, 0.016797048971056938f, 0.011110573075711727f, 0.015116985887289047f, 0.015084105543792248f, 0.012724609114229679f, 0.014323638752102852f, 0.011399950832128525f, 0.01516817044466734f, 0.012686953879892826f, 0.015723032876849174f, 0.013273685239255428f, 0.010650767013430595f, 0.013871495611965656f, 0.012070736847817898f, 0.01384354941546917f, 0.01187123078852892f, 0.013861080631613731f, 0.016597626730799675f, 0.014966469258069992f, 0.016096677631139755f, 0.015364019200205803f, 0.011497322469949722f);
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16 = 12;

static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16 = 12;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8 = -19;
static const ai_i8 _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.001747139380313456f;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32 = 9.565056097926572e-05f;
static const ai_float _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.000472932995762676f, 0.00036658989847637713f, 0.00044794526183977723f, 0.0004966641427017748f, 0.0003885256010107696f, 0.0003064904594793916f, 0.00043616938637569547f, 0.00043373170774430037f, 0.0004697151016443968f, 0.0003870527434628457f, 0.0003712075122166425f, 0.0005283931386657059f, 0.00037544549559243023f, 0.0004284371098037809f, 0.00042000433313660324f, 0.0004273742961231619f, 0.00041510979644954205f, 0.00042406850843690336f, 0.00047121255192905664f, 0.00042070061317645013f, 0.00033301394432783127f, 0.00043091317638754845f, 0.00035655309329740703f, 0.00040392691153101623f, 0.0004028857219964266f, 0.0004961087834089994f, 0.0003851308429148048f, 0.0004648486792575568f, 0.0003112196282017976f, 0.00041700710426084697f, 0.00039598895818926394f, 0.0004526131378952414f, 0.0003784341097343713f, 0.0004726302868220955f, 0.0004049896670039743f, 0.00034612944000400603f, 0.0003469349176157266f, 0.0003751476760953665f, 0.0004709293134510517f, 0.00036632962292060256f, 0.0003592880384530872f, 0.000485075666802004f, 0.0003483245673123747f, 0.0004097932833246887f, 0.000527844182215631f, 0.00033469978370703757f, 0.0004987673019059002f, 0.0004774696135427803f);
static const ai_layer_format_type _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN input_Transpose */
  {
    
  forward_lite_transpose_input_Transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END input_Transpose */
  /* LITE_KERNEL_SECTION BEGIN _stem_conv0_stem_Conv_output_0 */
  {
      const ai_i8* _stem_conv0_stem_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 28080);
    const ai_i8* _stem_conv0_stem_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    const ai_i32* _stem_conv0_stem_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 432);
    ai_i8* _stem_conv0_stem_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 18080);
    ai_i16* _stem_conv0_stem_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(137, 1, {(stai_ptr) _stem_conv0_stem_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_rgb_sssa8_ch(_stem_conv0_stem_Conv_output_0_t_in_0_ptr_const_s8, _stem_conv0_stem_Conv_output_0_t_in_0_shape_w_const_u16, _stem_conv0_stem_Conv_output_0_t_weight_0_ptr_const_s8, _stem_conv0_stem_Conv_output_0_t_out_0_shape_ch_const_u16, _stem_conv0_stem_Conv_output_0_t_weight_0_shape_w_const_u16, _stem_conv0_stem_Conv_output_0_l_pad_W_0_const_s32, _stem_conv0_stem_Conv_output_0_l_stride_0_const_u16, _stem_conv0_stem_Conv_output_0_t_weight_1_ptr_const_s32, _stem_conv0_stem_Conv_output_0_t_in_0_fmt_zero_const_s8, _stem_conv0_stem_Conv_output_0_t_out_0_fmt_zero_const_s8, _stem_conv0_stem_Conv_output_0_t_in_0_fmt_scale_const_f32, _stem_conv0_stem_Conv_output_0_t_out_0_fmt_scale_const_f32, _stem_conv0_stem_Conv_output_0_t_weight_0_fmt_scale_const_f32, _stem_conv0_stem_Conv_output_0_l_out_ch_format_const_layer_format_type, _stem_conv0_stem_Conv_output_0_t_out_0_ptr_s8, _stem_conv0_stem_Conv_output_0_t_out_0_shape_w_const_u16, 1196, _stem_conv0_stem_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(137, 1, {(stai_ptr) _stem_conv0_stem_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _stem_conv0_stem_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _stem_conv1_stem_Conv_output_0 */
  {
      const ai_i8* _stem_conv1_stem_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 18080);
    const ai_i8* _stem_conv1_stem_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 496);
    const ai_i32* _stem_conv1_stem_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 752);
    ai_i8* _stem_conv1_stem_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 17312);
    ai_i16* _stem_conv1_stem_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(140, 1, {(stai_ptr) _stem_conv1_stem_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_stem_conv1_stem_Conv_output_0_t_in_0_ptr_const_s8, _stem_conv1_stem_Conv_output_0_t_in_0_shape_w_const_u16, _stem_conv1_stem_Conv_output_0_t_in_0_shape_h_const_u16, _stem_conv1_stem_Conv_output_0_l_stride_1_const_u16, _stem_conv1_stem_Conv_output_0_l_stride_0_const_u16, _stem_conv1_stem_Conv_output_0_t_in_0_shape_ch_const_u16, _stem_conv1_stem_Conv_output_0_t_weight_0_ptr_const_s8, _stem_conv1_stem_Conv_output_0_t_out_0_shape_ch_const_u16, _stem_conv1_stem_Conv_output_0_t_weight_1_ptr_const_s32, _stem_conv1_stem_Conv_output_0_t_in_0_fmt_zero_const_s8, _stem_conv1_stem_Conv_output_0_t_out_0_fmt_zero_const_s8, _stem_conv1_stem_Conv_output_0_t_in_0_fmt_scale_const_f32, _stem_conv1_stem_Conv_output_0_t_out_0_fmt_scale_const_f32, _stem_conv1_stem_Conv_output_0_t_weight_0_fmt_scale_const_f32, _stem_conv1_stem_Conv_output_0_l_out_ch_format_const_layer_format_type, _stem_conv1_stem_Conv_output_0_t_out_0_ptr_s8, 1, 224, _stem_conv1_stem_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(140, 1, {(stai_ptr) _stem_conv1_stem_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _stem_conv1_stem_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 17312);
    const ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 816);
    const ai_i32* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1008);
    ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 1048);
    ai_i16* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(167, 1, {(stai_ptr) _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(_choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_w_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_h_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_pad_W_0_const_s32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_pad_H_0_const_s32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_w_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_h_const_u16, 1, 616, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(167, 1, {(stai_ptr) _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  {
      const ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1048);
    const ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1056);
    const ai_i32* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1356);
    ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8984);
    ai_i16* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7960);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(242, 1, {(stai_ptr) _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(_choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16, 0, 1021, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(242, 1, {(stai_ptr) _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8984);
    const ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1404);
    const ai_i32* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1548);
    ai_i8* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 600);
    ai_i16* _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(317, 1, {(stai_ptr) _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8, 1, 168, _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(317, 1, {(stai_ptr) _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split0_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split0_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7512);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 11180);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 9816);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(241, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(241, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 11180);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2252);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7696);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(316, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(316, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split1_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split1_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split1_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7984);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 13188);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 11824);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(240, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(240, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13188);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2300);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7984);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(315, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(315, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split2_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split2_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split2_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8272);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 13476);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12112);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(239, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(239, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13476);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2348);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8272);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(314, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(314, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split3_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split3_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split3_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8560);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 13764);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12400);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(238, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(238, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13764);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2396);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8560);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(313, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(313, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split4_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split4_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split4_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8848);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14052);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12688);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(237, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(237, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14052);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2444);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8848);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(312, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(312, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split5_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split5_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split5_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9136);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14340);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12976);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(236, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(236, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14340);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2492);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9136);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(311, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(311, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split6_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split6_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split6_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9424);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14628);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 13264);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(235, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(235, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14628);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2540);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9424);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(310, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(310, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split7_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split7_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split7_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14916);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 13552);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(234, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(234, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14916);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2588);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(309, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(309, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split8_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split8_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split8_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10000);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 15204);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 13840);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(233, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(233, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 15204);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2636);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10000);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(308, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(308, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split9_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split9_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split9_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10288);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 15492);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 14128);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(232, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(232, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 15492);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2684);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10288);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(307, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(307, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out */
  {
    
  forward_lite_concat_g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split10_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split10_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split10_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8876);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(231, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(231, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8876);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2732);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7696);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(306, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(306, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split11_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split11_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split11_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9348);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7984);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(230, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(230, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9348);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2780);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7984);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(305, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(305, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split12_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split12_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split12_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9636);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8272);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(229, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(229, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s12_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9636);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2828);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8272);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(304, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(304, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s12_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split13_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split13_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split13_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8560);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(228, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(228, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s13_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2876);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8560);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(303, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(303, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s13_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split14_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split14_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split14_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8848);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(227, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(227, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s14_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2924);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8848);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(302, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(302, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s14_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split15_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split15_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split15_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 9136);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(226, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(226, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s15_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2972);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9136);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(301, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(301, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s15_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split16_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split16_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split16_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9824);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 54368);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(225, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(225, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s16_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9824);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3020);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9424);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 55544);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(300, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(300, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s16_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split17_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split17_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split17_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13472);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 54368);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(224, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(224, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s17_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3068);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 55544);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(299, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(299, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s17_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split18_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split18_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split18_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13472);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 54464);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(223, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(223, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s18_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3116);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 54464);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(298, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(298, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s18_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split19_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split19_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split19_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 13472);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 54752);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(222, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(222, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s19_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9712);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3164);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 13456);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(297, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(297, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s19_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out */
  {
    
  forward_lite_concat_g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split20_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split20_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split20_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8876);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(221, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(221, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s20_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8876);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3212);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7696);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(296, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(296, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s20_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split21_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split21_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split21_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9348);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7984);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(220, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(220, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s21_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9348);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3260);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7984);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(295, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(295, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s21_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split22_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split22_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split22_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9636);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8272);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(219, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(219, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s22_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9636);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3308);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8272);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(294, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(294, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s22_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___stem_conv1_stem_Conv_output_0_split23_out */
  {
    
  forward_lite_slice_g2_5___stem_conv1_stem_Conv_output_0_split23_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___stem_conv1_stem_Conv_output_0_split23_out */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 54176);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1596);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1996);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16624);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8560);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(218, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_pad_W_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_pad_H_0_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_ptr_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_shape_h_const_u16, 0, 1361, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(218, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s23_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0 */
  {
      const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16624);
    const ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2060);
    const ai_i32* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3356);
    ai_i8* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8560);
    ai_i16* g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7512);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(293, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_shape_w_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_shape_h_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_l_stride_1_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_l_stride_0_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_weight_0_ptr_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_shape_ch_const_u16, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_weight_1_ptr_const_s32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_fmt_zero_const_s8, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_in_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_weight_0_fmt_scale_const_f32, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_l_out_ch_format_const_layer_format_type, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_ptr_s8, 1, 184, g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(293, 1, {(stai_ptr) g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s23_0 */
  /* LITE_KERNEL_SECTION BEGIN g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out */
  {
    
  forward_lite_concat_g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g2_5___choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g2_out */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0 */
  {
    
  forward_lite_concat__choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_0_Concat_output_0 */
  {
    
  forward_lite_concat__choice_blocks_0__layers_0_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_0_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_0_Transpose_output_0 */
  {
    
  forward_lite_transpose__choice_blocks_0__layers_0_Transpose_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_0_Transpose_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 432);
    const ai_i8* _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 3404);
    const ai_i32* _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3692);
    ai_i8* _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14472);
    ai_i16* _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 14256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(393, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 216, _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(393, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 432);
    const ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 3740);
    const ai_i32* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4028);
    ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 21384);
    ai_i16* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 14256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(392, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 216, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(392, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  {
      const ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 21384);
    const ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4076);
    const ai_i32* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4376);
    ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 1456);
    ai_i16* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(398, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(_choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16, 0, 1021, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(398, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1456);
    const ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4424);
    const ai_i32* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4568);
    ai_i8* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 21384);
    ai_i16* _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(401, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8, 1, 168, _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(401, 1, {(stai_ptr) _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_1_Concat_output_0 */
  {
    
  forward_lite_concat__choice_blocks_0__layers_1_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_1_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_0__layers_1_Transpose_output_0 */
  {
    
  forward_lite_transpose__choice_blocks_0__layers_1_Transpose_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_0__layers_1_Transpose_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14256);
    const ai_i8* _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4616);
    const ai_i32* _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 5192);
    ai_i8* _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 28080);
    ai_i16* _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(417, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 336, _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(417, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14256);
    const ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 5288);
    const ai_i32* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 5864);
    ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 41904);
    ai_i16* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(416, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 336, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(416, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before */
  {
      const ai_ptr _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 41904);
    ai_ptr _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(422, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_v_pad_constant_value_const_s8), _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(576), (ai_i32)(624), (ai_i32)(624), (ai_i32)(24), (ai_i32)(24));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(422, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 432);
    const ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 5960);
    const ai_i32* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 6176);
    ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 44192);
    ai_i16* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 16656);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(422, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(_choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16, 0, 889, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(422, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 44192);
    const ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 6272);
    const ai_i32* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 6848);
    ai_i8* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 43616);
    ai_i16* _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(425, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8, 1, 336, _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(425, 1, {(stai_ptr) _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_0_Concat_output_0 */
  {
    
  forward_lite_concat__choice_blocks_1__layers_0_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_0_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_0_Transpose_output_0 */
  {
    
  forward_lite_transpose__choice_blocks_1__layers_0_Transpose_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_0_Transpose_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 28080);
    const ai_i8* _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 6944);
    const ai_i32* _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 8096);
    ai_i8* _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 864);
    ai_i16* _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(441, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 432, _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(441, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 28080);
    const ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 8192);
    const ai_i32* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 9344);
    ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 27504);
    ai_i16* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(440, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 432, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(440, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before */
  {
      const ai_ptr _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 27504);
    ai_ptr _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 41328);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(446, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_v_pad_constant_value_const_s8), _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(576), (ai_i32)(624), (ai_i32)(624), (ai_i32)(24), (ai_i32)(24));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(446, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 41328);
    const ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 9440);
    const ai_i32* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 9656);
    ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 15580);
    ai_i16* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 14688);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(446, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(_choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16, 0, 889, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(446, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 15580);
    const ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 9752);
    const ai_i32* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 10328);
    ai_i8* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 15004);
    ai_i16* _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(449, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8, 1, 336, _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(449, 1, {(stai_ptr) _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_1_Concat_output_0 */
  {
    
  forward_lite_concat__choice_blocks_1__layers_1_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_1_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_1_Transpose_output_0 */
  {
    
  forward_lite_transpose__choice_blocks_1__layers_1_Transpose_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_1_Transpose_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 432);
    const ai_i8* _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 10424);
    const ai_i32* _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 11576);
    ai_i8* _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 28512);
    ai_i16* _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 28080);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(465, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 432, _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(465, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_2_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 432);
    const ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 11672);
    const ai_i32* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 12824);
    ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 42336);
    ai_i16* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 28080);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(464, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 432, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(464, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before */
  {
      const ai_ptr _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 42336);
    ai_ptr _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(470, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_ptr_const_ptr, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_v_pad_constant_value_const_s8), _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(576), (ai_i32)(624), (ai_i32)(624), (ai_i32)(24), (ai_i32)(24));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(470, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 432);
    const ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 12920);
    const ai_i32* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 13136);
    ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 44192);
    ai_i16* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 16656);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(470, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(_choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16, 0, 889, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(470, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_2_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 44192);
    const ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 13232);
    const ai_i32* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 13808);
    ai_i8* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 43616);
    ai_i16* _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(473, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8, 1, 336, _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(473, 1, {(stai_ptr) _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_2_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_2_Concat_output_0 */
  {
    
  forward_lite_concat__choice_blocks_1__layers_2_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_2_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_1__layers_2_Transpose_output_0 */
  {
    
  forward_lite_transpose__choice_blocks_1__layers_2_Transpose_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_1__layers_2_Transpose_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 28080);
    const ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 13904);
    const ai_i32* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 16208);
    ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5904);
    ai_i16* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(500, 1, {(stai_ptr) _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(_choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_w_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_shape_h_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_w_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_h_const_u16, 1, 1, 5472, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(500, 1, {(stai_ptr) _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  {
      const ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5904);
    const ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 16400);
    const ai_i32* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 18752);
    ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 20356);
    ai_i16* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12816);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(539, 1, {(stai_ptr) _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(_choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16, 0, 7537, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(539, 1, {(stai_ptr) _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_0_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 20356);
    const ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 18944);
    const ai_i32* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 21248);
    ai_i8* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 1104);
    ai_i16* _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(578, 1, {(stai_ptr) _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8, 1, 672, _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(578, 1, {(stai_ptr) _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_0_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split0_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8016);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 20164);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12624);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(538, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(538, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s0_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 20164);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 26288);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8016);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(577, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(577, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s0_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split1_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8592);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 23044);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 15504);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(537, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(537, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s1_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 23044);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 26480);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8592);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(576, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(576, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s1_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split2_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9168);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 24772);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 17232);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(536, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(536, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s2_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 24772);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 26672);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9168);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(575, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(575, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s2_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split3_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9744);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 25348);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 17808);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(535, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(535, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s3_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 25348);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 26864);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9744);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(574, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(574, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s3_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split4_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10320);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 25924);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 18384);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(534, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(534, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s4_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 25924);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 27056);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10320);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(573, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(573, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s4_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split5_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10896);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 26500);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 18960);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(533, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(533, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s5_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 26500);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 27248);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10896);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(572, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(572, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s5_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split6_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 11472);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 55728);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 19536);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(532, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(532, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s6_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 55728);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 27440);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 11472);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(571, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(571, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s6_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split7_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 12048);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 55728);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 20112);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(531, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(531, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s7_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 55728);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 27632);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 56832);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(570, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(570, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s7_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split8_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 12048);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 55728);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 20112);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(530, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(530, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s8_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 55728);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 27824);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 57408);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(569, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(569, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s8_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split9_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 20016);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12476);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(529, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(529, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s9_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 28016);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 57984);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12048);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(568, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(568, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s9_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out */
  {
    
  forward_lite_concat_g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g0_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split10_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14256);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 55728);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(528, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(528, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s10_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 28208);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8688);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8016);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(567, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(567, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s10_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out */
  {
    
  forward_lite_slice_g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_1__layers_2_Reshape_1_output_0_split11_out */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9264);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21440);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23792);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 28080);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(527, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_W_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_pad_H_0_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_ptr_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_shape_h_const_u16, 0, 7537, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(527, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu1_Relu_output_0_split_s11_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0 */
  {
      const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23984);
    const ai_i32* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 28400);
    ai_i8* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9264);
    ai_i16* g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8016);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(566, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_w_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_h_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_1_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_stride_0_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_ptr_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_shape_ch_const_u16, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_1_ptr_const_s32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_zero_const_s8, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_in_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_weight_0_fmt_scale_const_f32, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_l_out_ch_format_const_layer_format_type, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_ptr_s8, 1, 672, g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(566, 1, {(stai_ptr) g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_split_s11_0 */
  /* LITE_KERNEL_SECTION BEGIN g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out */
  {
    
  forward_lite_concat_g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END g59_62___choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0_concat_s0_g1_out */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0 */
  {
    
  forward_lite_concat__choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_0_branch1_shufflebk_branch1_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_0_Concat_output_0 */
  {
    
  forward_lite_concat__choice_blocks_2__layers_0_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_0_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_0_Transpose_output_0 */
  {
    
  forward_lite_transpose__choice_blocks_2__layers_0_Transpose_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_0_Transpose_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 28592);
    const ai_i32* _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 33200);
    ai_i8* _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14688);
    ai_i16* _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 13824);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(627, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 864, _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(627, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_1_branch1_shufflebk_branch1_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 33392);
    const ai_i32* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 38000);
    ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 21600);
    ai_i16* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 13824);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(626, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_1_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_stride_0_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8, 1, 864, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(626, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  {
      const ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 21600);
    const ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 38192);
    const ai_i32* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 40544);
    ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7540);
    ai_i16* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(632, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_dw_sssa8_ch(_choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_ptr_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_w_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_h_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_ptr_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_w_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_shape_h_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_W_0_const_s32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_pad_H_0_const_s32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_1_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_l_stride_0_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_1_ptr_const_s32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_w_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_shape_h_const_u16, 0, 7537, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(632, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_1_branch2_shufflebk_branch2_conv2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  {
      const ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7540);
    const ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 40736);
    const ai_i32* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 43040);
    ai_i8* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 21600);
    ai_i16* _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(635, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_ptr_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_w_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_h_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_1_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_stride_0_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_shape_ch_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_ptr_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_shape_ch_const_u16, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_1_ptr_const_s32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_zero_const_s8, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_in_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_weight_0_fmt_scale_const_f32, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_l_out_ch_format_const_layer_format_type, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8, 1, 672, _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(635, 1, {(stai_ptr) _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_1_branch2_shufflebk_branch2_relu2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_1_Concat_output_0 */
  {
    
  forward_lite_concat__choice_blocks_2__layers_1_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_1_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _choice_blocks_2__layers_1_Transpose_output_0 */
  {
    
  forward_lite_transpose__choice_blocks_2__layers_1_Transpose_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _choice_blocks_2__layers_1_Transpose_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _global_pooling_GlobalAveragePool_output_0 */
  {
    
  forward_lite_ap_integer_INT8__global_pooling_GlobalAveragePool_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _global_pooling_GlobalAveragePool_output_0 */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input */
  {
    
  forward_lite_dense_integer_SSSA_ch_output_QuantizeLinear_Input(net_ctx);
  }
  /* LITE_KERNEL_SECTION END output_QuantizeLinear_Input */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 432;

  net_ctx->_outputs[0] = activations[0] + 1288;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

