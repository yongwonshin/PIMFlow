import onnx
import math
import numpy as np
import onnx.numpy_helper as numpy_helper
from google.protobuf.json_format import MessageToDict
from pim.util import find_initializer_by_arg_name, find_attribute_by_name, get_arg_shape, find_nodes_by_arg_name, create_initializer_tensor, gvn, find_node_index_by_name, onnx_datatype, silu_like_type, supportedActivationNodes, par_exec_id

class TransformerBase:
  def reset(self):
    raise Exception("Not implemented!")

  def satisfy(self, graph, node):
    return False

  def apply(self, graph, node):
    pass

  def transform(self, model, nodes, stage=2, is_gpu_first=True):
    if self.satisfy(model.graph, nodes):
      for i in range(len(nodes)):
        self.apply(model.graph, nodes, stage, i, is_gpu_first)
        model = onnx.shape_inference.infer_shapes(model)
      self.reset()
    return model

class InputSplit(TransformerBase):
  def __init__(self, ratio, nodes={}, onnx_datatype=onnx_datatype, node_map={}):
    self.split_ratio = ratio
    self.ONNX_DATATYPE = onnx_datatype
    self.nodes = nodes
    self.node_map = node_map

  def reset(self):
    pass

  def satisfy(self, graph, node):
    if len(self.nodes) > 0:
      ratio = self.nodes.get(node.name, None)
      if ratio is None:
        return False
      else:
        self.split_ratio = ratio / 100

    if node.op_type == "Conv":
      # already processed
      if node.name.endswith("_added"):
        return False

      # depth-wise conv
      weight = find_initializer_by_arg_name(graph, node.input[1])
      group = find_attribute_by_name(node, 'group').i
      if weight.dims[1] == 1 and group == weight.dims[0] or group == 32:
        return False

      # split has no effect when H-dim == 1
      if get_arg_shape(graph, node, node.input[0])[2] == 1:
        return False

      # fc
      h = get_arg_shape(graph, node, node.input[0])[2]
      w = get_arg_shape(graph, node, node.input[0])[3]
      if h == 1 and w == 1 and weight.dims[2] == 1 and weight.dims[3] == 1:
        return False

      # TODO: now ignore dilation > 1
      # TODO: stride > 1 may have potential bugs
      dilations = find_attribute_by_name(node, 'dilations').ints
      for dilation in dilations:
        if dilation > 1:
          return False

      h_gpu = self.get_h_gpu(graph, node)
      kernel_shape = find_attribute_by_name(node, 'kernel_shape').ints
      strides = find_attribute_by_name(node, 'strides').ints
      ps_gpu = (h_gpu - 1) % strides[0]
      # no need to split
      if self.get_h_gpu(graph, node) + kernel_shape[0] // 2 - ps_gpu >= h - 1:
        return False

      # only gpu
      if math.isclose(self.split_ratio, 1):
        return False

      return True
    else:
      return False

  def get_h_gpu(self, graph, node):
    return math.ceil(get_arg_shape(graph, node, node.input[0])[2] * self.split_ratio)

  def apply(self, graph, node):
    # total offloading
    if math.isclose(self.split_ratio, 0):
      conv_output = f"token_{gvn()}"
      new_name = f'Conv_{gvn()}_{par_exec_id()}_pim_added'
      conv = onnx.helper.make_node(
        'Conv',
        name=new_name,
        inputs=node.input,
        outputs=[conv_output],
        dilations=find_attribute_by_name(node, 'dilations').ints,
        group=find_attribute_by_name(node, 'group').i,
        kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
        pads=find_attribute_by_name(node, 'pads').ints,
        strides=find_attribute_by_name(node, 'strides').ints,
      )
      n = find_node_index_by_name(graph, node.name)
      graph.node.insert(n, conv)

      next_nodes = find_nodes_by_arg_name(graph, node.output[0])
      next_nodes.remove(node)
      for n in next_nodes:
        for i in range(len(n.input)):
          if n.input[i] == node.output[0]:
            n.input[i] = conv_output

      graph.node.remove(node)
      return

    weight = find_initializer_by_arg_name(graph, node.input[1])
    bias = None
    if len(node.input) > 2:
      bias = find_initializer_by_arg_name(graph, node.input[2])

    # add split node (gpu)
    input_arg = node.input[0]
    if graph.node[0] != node:
      input_nodes = find_nodes_by_arg_name(graph, node.input[0])
      input_nodes.remove(node)
      input_arg = input_nodes[0].output[0]

    pads = find_attribute_by_name(node, 'pads').ints
    strides = find_attribute_by_name(node, 'strides').ints
    assert pads[0] == pads[2]

    h = get_arg_shape(graph, node, node.input[0])[2]
    kernel_shape = find_attribute_by_name(node, 'kernel_shape').ints
    h_gpu = self.get_h_gpu(graph, node)

    conv_arg = f"token_{gvn()}"
    pads_name = f"token_{gvn()}"
    tensor_ = create_initializer_tensor(
      name=pads_name,
      tensor_array=np.array([0, 0, pads[0], 0, 0, 0, pads[2], 0]),
      data_type=onnx.TensorProto.INT64)
    graph.initializer.append(tensor_)
    pad_node_gpu = onnx.helper.make_node(
      'Pad',
      name=f'Pad_{gvn()}',
      inputs=[input_arg, pads_name],
      outputs=[conv_arg],
    )
    n = find_node_index_by_name(graph, node.name)
    graph.node.insert(n, pad_node_gpu)

    slice_start_name = f"token_{gvn()}"
    tensor_ = create_initializer_tensor(
      name=slice_start_name,
      tensor_array=np.array([0]),
      data_type=onnx.TensorProto.INT64)
    graph.initializer.append(tensor_)

    ps_gpu = (h_gpu - 1) % strides[0]
    ps_pim = (-h_gpu) % strides[0]

    slice_end_name = f"token_{gvn()}"
    tensor_ = create_initializer_tensor(
      name=slice_end_name,
      tensor_array=np.array([h_gpu + pads[0] + kernel_shape[0] // 2 - ps_gpu]),
      data_type=onnx.TensorProto.INT64)
    graph.initializer.append(tensor_)

    slice_axis_name = f"token_{gvn()}"
    tensor_ = create_initializer_tensor(
      name=slice_axis_name,
      tensor_array=np.array([2]),
      data_type=onnx.TensorProto.INT64)
    graph.initializer.append(tensor_)

    split_output1 = f"token_{gvn()}"
    split_node_gpu = onnx.helper.make_node(
      'Slice',
      name=f'Slice_{gvn()}_',
      inputs=[conv_arg, slice_start_name, slice_end_name, slice_axis_name],
      outputs=[split_output1],
    )
    graph.node.insert(n+1, split_node_gpu)

    # add split node (pim)
    slice_start_name = f"token_{gvn()}"
    tensor_ = create_initializer_tensor(
      name=slice_start_name,
      tensor_array=np.array([h_gpu + pads[0] - kernel_shape[0] // 2 + ps_pim]),
      data_type=onnx.TensorProto.INT64)
    graph.initializer.append(tensor_)

    slice_end_name = f"token_{gvn()}"
    tensor_ = create_initializer_tensor(
      name=slice_end_name,
      tensor_array=np.array([h + pads[0] + pads[2]]),
      data_type=onnx.TensorProto.INT64)
    graph.initializer.append(tensor_)

    slice_axis_name = f"token_{gvn()}"
    tensor_ = create_initializer_tensor(
      name=slice_axis_name,
      tensor_array=np.array([2]),
      data_type=onnx.TensorProto.INT64)
    graph.initializer.append(tensor_)

    split_output2 = f"token_{gvn()}"
    split_node_pim = onnx.helper.make_node(
      'Slice',
      name=f'Slice_{gvn()}_',
      inputs=[conv_arg, slice_start_name, slice_end_name, slice_axis_name],
      outputs=[split_output2],
    )
    graph.node.insert(n+2, split_node_pim)

    # add conv (gpu) node
    conv_gpu_weight_name = f"token_{gvn()}"
    conv_gpu_weight = create_initializer_tensor(
      name=conv_gpu_weight_name,
      tensor_array=numpy_helper.to_array(weight), # use the same weight
      data_type=self.ONNX_DATATYPE)
    graph.initializer.append(conv_gpu_weight)

    conv_gpu_bias_name = None
    if bias is not None:
      conv_gpu_bias_name = f"token_{gvn()}"
      conv_gpu_bias = create_initializer_tensor(
        name=conv_gpu_bias_name,
        tensor_array=numpy_helper.to_array(bias),
        data_type=self.ONNX_DATATYPE)
      graph.initializer.append(conv_gpu_bias)

    inputs = [split_output1, conv_gpu_weight_name]
    if bias is not None:
      inputs.append(conv_gpu_bias_name)

    conv_gpu_output = f"token_{gvn()}"
    new_name = f'Conv_{gvn()}_{par_exec_id()}_added'
    conv_gpu = onnx.helper.make_node(
      'Conv',
      name=new_name,
      inputs=inputs,
      outputs=[conv_gpu_output],
      dilations=find_attribute_by_name(node, 'dilations').ints,
      group=find_attribute_by_name(node, 'group').i,
      kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
      pads=[0, pads[1], 0, pads[3]],
      strides=find_attribute_by_name(node, 'strides').ints,
    )
    graph.node.insert(n+3, conv_gpu)
    self.node_map[node.name] = [new_name]

    # add conv (pim) node
    conv_pim_weight_name = f"token_{gvn()}"
    conv_pim_weight = create_initializer_tensor(
      name=conv_pim_weight_name,
      tensor_array=numpy_helper.to_array(weight), # use the same weight
      data_type=self.ONNX_DATATYPE)
    graph.initializer.append(conv_pim_weight)

    conv_pim_bias_name = None
    if bias is not None:
      conv_pim_bias_name = f"token_{gvn()}"
      conv_pim_bias = create_initializer_tensor(
        name=conv_pim_bias_name,
        tensor_array=numpy_helper.to_array(bias),
        data_type=self.ONNX_DATATYPE)
      graph.initializer.append(conv_pim_bias)

    inputs = [split_output2, conv_pim_weight_name]
    if bias is not None:
      inputs.append(conv_pim_bias_name)

    conv_pim_output = f"token_{gvn()}"
    conv_pim = onnx.helper.make_node(
      'Conv',
      name=f'Conv_{gvn()}_{par_exec_id()}_pim_added',
      inputs=inputs,
      outputs=[conv_pim_output],
      dilations=find_attribute_by_name(node, 'dilations').ints,
      group=find_attribute_by_name(node, 'group').i,
      kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
      pads=[0, pads[1], 0, pads[3]],
      strides=find_attribute_by_name(node, 'strides').ints,
    )
    graph.node.insert(n+4, conv_pim)

    outputs = [conv_gpu_output, conv_pim_output]
    next_nodes = find_nodes_by_arg_name(graph, node.output[0])
    next_nodes.remove(node)

    # last activation
    prev_node = node
    nodes_to_remove = []
    while supportedActivationNodes(graph, next_nodes):
      print([node.name for node in next_nodes])
      nodes_to_remove.extend(next_nodes)

      if len(next_nodes) == 1:
        # add element-wise nodes
        n = find_node_index_by_name(graph, next_nodes[0].name)
        outputs_new = []
        for i in range(2):
          ew_node_output = f"token_{gvn()}"
          ew_node = onnx.helper.make_node(
            next_nodes[0].op_type,
            name=f"{next_nodes[0].op_type}_{gvn()}_",
            inputs=[outputs[i]] + next_nodes[0].input[1:],
            outputs=[ew_node_output],
          )
          graph.node.insert(n + i, ew_node)
          outputs_new.append(ew_node_output)
        outputs = outputs_new
      elif len(next_nodes) == 2:
        n = find_node_index_by_name(graph, next_nodes[-1].name)
        outputs_new = []
        for i in range(2):
          sigmoid_node_output = f"token_{gvn()}"
          sigmoid_node = onnx.helper.make_node(
            silu_like_type(graph, next_nodes),
            name=f"Sigmoid_{gvn()}_",
            inputs=[outputs[i]],
            outputs=[sigmoid_node_output],
          )
          graph.node.insert(n + 2*i, sigmoid_node)

          mul_node_output = f"token_{gvn()}"
          mul_node = onnx.helper.make_node(
            "Mul",
            name=f"Mul_{gvn()}_",
            inputs=[outputs[i], sigmoid_node_output],
            outputs=[mul_node_output],
          )
          graph.node.insert(n + 2*i + 1, mul_node)
          outputs_new.append(mul_node_output)
        outputs = outputs_new

      prev_node = next_nodes[-1]

      next_nodes_new = find_nodes_by_arg_name(graph, next_nodes[-1].output[0])
      next_nodes_new.remove(next_nodes[-1])
      next_nodes = next_nodes_new

    # add concat node
    concat_node_output = f"token_{gvn()}"
    concat_node = onnx.helper.make_node(
      'Concat',
      name=f'Concat_{gvn()}_',
      inputs=outputs,
      outputs=[concat_node_output],
      axis=2
    )
    n = max([max([
      find_node_index_by_name(graph, node.name)
        for node in find_nodes_by_arg_name(graph, output)])
          for output in outputs])
    graph.node.insert(n+1, concat_node)

    for n in next_nodes:
      for i in range(len(n.input)):
        if n.input[i] == prev_node.output[0]:
          n.input[i] = concat_node_output

    graph.node.remove(node)
    for n in nodes_to_remove:
      graph.node.remove(n)

  def transform(self, model):
    while True:
      inferred_model = None
      for node in model.graph.node:
        print(node.name)
        if self.satisfy(model.graph, node):
          self.apply(model.graph, node)
          self.reset()
          print(MessageToDict(node))
          inferred_model = onnx.shape_inference.infer_shapes(model)
          par_exec_id(True)
          break

      if inferred_model is None:
        break

      model = inferred_model
    return model


# TODO: MaxPool (vgg-16)
# TODO: Add (resnet-18)
# TODO: Mul (efficientnet)
# NOTE: GlobalAveragePool is not suppored (reduction operators)
class Pipeline(TransformerBase):
  def __init__(self, onnx_datatype=onnx_datatype, node_map={}):
    self.prev_conv2 = None
    self.extra_slice = 0
    self.outputs = []
    self.ONNX_DATATYPE = onnx_datatype
    self.node_map = node_map

  def reset(self):
    self.prev_conv2 = None
    self.extra_slice = 0
    self.outputs = []

  def satisfy(self, graph, nodes):
    return True

  def get_h_gpu(self, graph, node):
    return get_arg_shape(graph, node, node.input[0])[2] // 2

  def postfix(self, nodes, i, is_gpu_first):
    conv_nodes = filter(lambda x: x.startswith("Conv"), nodes[:i+1])
    n = len(list(conv_nodes))
    if n % 2 == 1 and is_gpu_first or n % 2 == 0 and not is_gpu_first:
      return "added"
    else:
      return "pim_added"

  def apply(self, graph, nodes, stage, i, is_gpu_first):
    nodes_to_remove = []
    if i == 0:
      node_index = find_node_index_by_name(graph, nodes[0])
      node = graph.node[node_index]

      weight = find_initializer_by_arg_name(graph, node.input[1])
      bias = None
      if len(node.input) > 2:
        bias = find_initializer_by_arg_name(graph, node.input[2])

      # add split node
      input_arg = node.input[0]
      if graph.node[0] != node:
        input_nodes = find_nodes_by_arg_name(graph, node.input[0])
        input_nodes.remove(node)
        input_arg = input_nodes[0].output[0]

      pads = find_attribute_by_name(node, 'pads').ints
      assert pads[0] == pads[2]

      h = get_arg_shape(graph, node, node.input[0])[2]
      kernel_shape = find_attribute_by_name(node, 'kernel_shape').ints
      h_gpu = self.get_h_gpu(graph, node)

      strides = find_attribute_by_name(node, 'strides').ints
      ps_gpu = (h_gpu - 1) % strides[0]
      ps_pim = (-h_gpu) % strides[0]

      conv1_arg = f"token_{gvn()}"

      slice_start_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_start_name,
        tensor_array=np.array([0]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      slice_end_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_end_name,
        tensor_array=np.array([h_gpu + kernel_shape[0] // 2 - ps_gpu]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      slice_axis_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_axis_name,
        tensor_array=np.array([2]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      split_output = f"token_{gvn()}"
      split_node = onnx.helper.make_node(
        'Slice',
        name=f'Slice_{gvn()}_',
        inputs=[input_arg, slice_start_name, slice_end_name, slice_axis_name],
        outputs=[split_output],
      )
      n = find_node_index_by_name(graph, node.name)
      graph.node.insert(n, split_node)

      pads_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=pads_name,
        tensor_array=np.array([0, 0, pads[0], 0, 0, 0, 0, 0]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)
      pad_node = onnx.helper.make_node(
        'Pad',
        name=f'Pad_{gvn()}_',
        inputs=[split_output, pads_name],
        outputs=[conv1_arg],
      )
      graph.node.insert(n+1, pad_node)

      # add split node
      conv2_arg = f"token_{gvn()}"

      slice_start_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_start_name,
        tensor_array=np.array([h_gpu - kernel_shape[0] // 2 + ps_pim]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      slice_end_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_end_name,
        tensor_array=np.array([h]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      slice_axis_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_axis_name,
        tensor_array=np.array([2]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      split_output = f"token_{gvn()}"
      split_node = onnx.helper.make_node(
        'Slice',
        name=f'Slice_{gvn()}_',
        inputs=[input_arg, slice_start_name, slice_end_name, slice_axis_name],
        outputs=[split_output],
      )
      graph.node.insert(n+2, split_node)

      pads_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=pads_name,
        tensor_array=np.array([0, 0, 0, 0, 0, 0, pads[2], 0]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)
      pad_node = onnx.helper.make_node(
        'Pad',
        name=f'Pad_{gvn()}_',
        inputs=[split_output, pads_name],
        outputs=[conv2_arg],
      )
      graph.node.insert(n+3, pad_node)

      # add conv node
      conv_weight_name = f"token_{gvn()}"
      conv_weight = create_initializer_tensor(
        name=conv_weight_name,
        tensor_array=numpy_helper.to_array(weight), # use the same weight
        data_type=self.ONNX_DATATYPE)
      graph.initializer.append(conv_weight)

      conv_bias_name = None
      if bias is not None:
        conv_bias_name = f"token_{gvn()}"
        conv_bias = create_initializer_tensor(
          name=conv_bias_name,
          tensor_array=numpy_helper.to_array(bias),
          data_type=self.ONNX_DATATYPE)
        graph.initializer.append(conv_bias)

      inputs = [conv1_arg, conv_weight_name]
      if bias is not None:
        inputs.append(conv_bias_name)

      conv1_output = f"token_{gvn()}"
      new_name = f'Conv_{gvn()}_{par_exec_id()}_{self.postfix(nodes, i, is_gpu_first)}'
      conv1 = onnx.helper.make_node(
        'Conv',
        name=new_name,
        inputs=inputs,
        outputs=[conv1_output],
        dilations=find_attribute_by_name(node, 'dilations').ints,
        group=find_attribute_by_name(node, 'group').i,
        kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
        pads=[0, pads[1], 0, pads[3]],
        strides=find_attribute_by_name(node, 'strides').ints,
      )
      graph.node.insert(n+4, conv1)
      if not new_name.endswith("pim_added") and new_name.endswith("added"):
        self.node_map[node.name] = [new_name]

      # add conv node
      conv_weight_name = f"token_{gvn()}"
      conv_weight = create_initializer_tensor(
        name=conv_weight_name,
        tensor_array=numpy_helper.to_array(weight), # use the same weight
        data_type=self.ONNX_DATATYPE)
      graph.initializer.append(conv_weight)

      conv_bias_name = None
      if bias is not None:
        conv_bias_name = f"token_{gvn()}"
        conv_bias = create_initializer_tensor(
          name=conv_bias_name,
          tensor_array=numpy_helper.to_array(bias),
          data_type=self.ONNX_DATATYPE)
        graph.initializer.append(conv_bias)

      inputs = [conv2_arg, conv_weight_name]
      if bias is not None:
        inputs.append(conv_bias_name)

      conv2_output = f"token_{gvn()}"
      new_name = f'Conv_{gvn()}_{par_exec_id(True)}_{self.postfix(nodes, i, is_gpu_first)}'
      conv2 = onnx.helper.make_node(
        'Conv',
        name=new_name,
        inputs=inputs,
        outputs=[conv2_output],
        dilations=find_attribute_by_name(node, 'dilations').ints,
        group=find_attribute_by_name(node, 'group').i,
        kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
        pads=[0, pads[1], 0, pads[3]],
        strides=find_attribute_by_name(node, 'strides').ints,
      )
      graph.node.insert(n+5, conv2)
      if not new_name.endswith("pim_added") and new_name.endswith("added"):
        self.node_map[node.name].append(new_name)

      outputs = [conv1_output, conv2_output]
      next_nodes = find_nodes_by_arg_name(graph, node.output[0])
      next_nodes.remove(node)
      self.prev_conv2 = conv2

      # last activation
      prev_node = node
      if supportedActivationNodes(graph, next_nodes):
        print([node.name for node in next_nodes])
        nodes_to_remove.extend(next_nodes)

        if len(next_nodes) == 1:
          # add element-wise nodes
          n = find_node_index_by_name(graph, next_nodes[0].name)
          outputs_new = []
          for i in range(2):
            ew_node_output = f"token_{gvn()}"
            ew_node = onnx.helper.make_node(
              next_nodes[0].op_type,
              name=f"{next_nodes[0].op_type}_{gvn()}_",
              inputs=[outputs[i]] + next_nodes[0].input[1:],
              outputs=[ew_node_output],
            )
            graph.node.insert(n + i, ew_node)
            outputs_new.append(ew_node_output)
            self.prev_conv2 = ew_node
          outputs = outputs_new
        elif len(next_nodes) == 2:
          n = find_node_index_by_name(graph, next_nodes[-1].name)
          outputs_new = []
          for i in range(2):
            sigmoid_node_output = f"token_{gvn()}"
            sigmoid_node = onnx.helper.make_node(
              silu_like_type(graph, next_nodes),
              name=f"Sigmoid_{gvn()}_",
              inputs=[outputs[i]],
              outputs=[sigmoid_node_output],
            )
            graph.node.insert(n + 2*i, sigmoid_node)

            mul_node_output = f"token_{gvn()}"
            mul_node = onnx.helper.make_node(
              "Mul",
              name=f"Mul_{gvn()}_",
              inputs=[outputs[i], sigmoid_node_output],
              outputs=[mul_node_output],
            )
            graph.node.insert(n + 2*i + 1, mul_node)
            outputs_new.append(mul_node_output)
            self.prev_conv2 = mul_node
          outputs = outputs_new

        prev_node = next_nodes[-1]

        next_nodes_new = find_nodes_by_arg_name(graph, next_nodes[-1].output[0])
        next_nodes_new.remove(next_nodes[-1])
        next_nodes = next_nodes_new

      # add concat node
      concat_node_output = f"token_{gvn()}"
      concat_node = onnx.helper.make_node(
        'Concat',
        name=f'Concat_{gvn()}_',
        inputs=outputs,
        outputs=[concat_node_output],
        axis=2
      )
      n = max([max([
        find_node_index_by_name(graph, node.name)
          for node in find_nodes_by_arg_name(graph, output)])
            for output in outputs])
      graph.node.insert(n+1, concat_node)

      for n in next_nodes:
        for i in range(len(n.input)):
          if n.input[i] == prev_node.output[0]:
            n.input[i] = concat_node_output

      graph.node.remove(node)
      self.outputs = outputs
    else:
      node_index = find_node_index_by_name(graph, nodes[i])
      node = graph.node[node_index]

      weight = find_initializer_by_arg_name(graph, node.input[1])
      bias = None
      if len(node.input) > 2:
        bias = find_initializer_by_arg_name(graph, node.input[2])

      # add split node
      input_arg = node.input[0]
      if graph.node[0] != node:
        input_nodes = find_nodes_by_arg_name(graph, node.input[0])
        input_nodes.remove(node)
        input_arg = input_nodes[0].output[0]

      pads = find_attribute_by_name(node, 'pads').ints
      assert pads[0] == pads[2]

      h = get_arg_shape(graph, node, node.input[0])[2]
      kernel_shape = find_attribute_by_name(node, 'kernel_shape').ints
      self.extra_slice += kernel_shape[0] // 2
      h_gpu = self.get_h_gpu(graph, node)

      strides = find_attribute_by_name(node, 'strides').ints
      ps_gpu = (h_gpu - 1) % strides[0]
      ps_pim = (-h_gpu) % strides[0]

      conv1_arg = f"token_{gvn()}"

      slice_start_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_start_name,
        tensor_array=np.array([0]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      slice_end_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_end_name,
        tensor_array=np.array([h_gpu + self.extra_slice + kernel_shape[0] // 2 - ps_gpu]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      slice_axis_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=slice_axis_name,
        tensor_array=np.array([2]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)

      split_output = f"token_{gvn()}"
      split_node = onnx.helper.make_node(
        'Slice',
        name=f'Slice_{gvn()}_',
        inputs=[input_arg, slice_start_name, slice_end_name, slice_axis_name],
        outputs=[split_output],
      )
      n = find_node_index_by_name(graph, node.name)
      graph.node.insert(n, split_node)

      pads_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=pads_name,
        tensor_array=np.array([0, 0, pads[0], 0, 0, 0, 0, 0]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)
      pad_node = onnx.helper.make_node(
        'Pad',
        name=f'Pad_{gvn()}_',
        inputs=[split_output, pads_name],
        outputs=[conv1_arg],
      )
      graph.node.insert(n+1, pad_node)

      conv2_arg = f"token_{gvn()}"

      pads_name = f"token_{gvn()}"
      tensor_ = create_initializer_tensor(
        name=pads_name,
        tensor_array=np.array([0, 0, 0, 0, 0, 0, pads[2], 0]),
        data_type=onnx.TensorProto.INT64)
      graph.initializer.append(tensor_)
      pad_node = onnx.helper.make_node(
        'Pad',
        name=f'Pad_{gvn()}_',
        inputs=[self.prev_conv2.output[0], pads_name],
        outputs=[conv2_arg],
      )
      graph.node.insert(n+2, pad_node)

      # add conv node
      conv_weight_name = f"token_{gvn()}"
      conv_weight = create_initializer_tensor(
        name=conv_weight_name,
        tensor_array=numpy_helper.to_array(weight), # use the same weight
        data_type=self.ONNX_DATATYPE)
      graph.initializer.append(conv_weight)

      conv_bias_name = None
      if bias is not None:
        conv_bias_name = f"token_{gvn()}"
        conv_bias = create_initializer_tensor(
          name=conv_bias_name,
          tensor_array=numpy_helper.to_array(bias),
          data_type=self.ONNX_DATATYPE)
        graph.initializer.append(conv_bias)

      inputs = [conv1_arg, conv_weight_name]
      if bias is not None:
        inputs.append(conv_bias_name)

      conv1_output = f"token_{gvn()}"
      new_name = f'Conv_{gvn()}_{par_exec_id()}_{self.postfix(nodes, i, is_gpu_first)}'
      conv1 = onnx.helper.make_node(
        'Conv',
        name=new_name,
        inputs=inputs,
        outputs=[conv1_output],
        dilations=find_attribute_by_name(node, 'dilations').ints,
        group=find_attribute_by_name(node, 'group').i,
        kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
        pads=[0, pads[1], 0, pads[3]],
        strides=find_attribute_by_name(node, 'strides').ints,
      )
      graph.node.insert(n+3, conv1)
      if not new_name.endswith("pim_added") and new_name.endswith("added"):
        self.node_map[node.name] = [new_name]

      # add conv node
      conv_weight_name = f"token_{gvn()}"
      conv_weight = create_initializer_tensor(
        name=conv_weight_name,
        tensor_array=numpy_helper.to_array(weight), # use the same weight
        data_type=self.ONNX_DATATYPE)
      graph.initializer.append(conv_weight)

      conv_bias_name = None
      if bias is not None:
        conv_bias_name = f"token_{gvn()}"
        conv_bias = create_initializer_tensor(
          name=conv_bias_name,
          tensor_array=numpy_helper.to_array(bias),
          data_type=self.ONNX_DATATYPE)
        graph.initializer.append(conv_bias)

      inputs = [conv2_arg, conv_weight_name]
      if bias is not None:
        inputs.append(conv_bias_name)

      conv2_output = f"token_{gvn()}"
      new_name = f'Conv_{gvn()}_{par_exec_id(True)}_{self.postfix(nodes, i, is_gpu_first)}'
      conv2 = onnx.helper.make_node(
        'Conv',
        name=new_name,
        inputs=inputs,
        outputs=[conv2_output],
        dilations=find_attribute_by_name(node, 'dilations').ints,
        group=find_attribute_by_name(node, 'group').i,
        kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
        pads=[0, pads[1], 0, pads[3]],
        strides=find_attribute_by_name(node, 'strides').ints,
      )
      graph.node.insert(n+4, conv2)
      if not new_name.endswith("pim_added") and new_name.endswith("added"):
        self.node_map[node.name].append(new_name)

      outputs = [conv1_output, conv2_output]
      next_nodes = find_nodes_by_arg_name(graph, node.output[0])
      next_nodes.remove(node)
      self.prev_conv2 = conv2

      # last activation
      prev_node = node
      if supportedActivationNodes(graph, next_nodes):
        print([node.name for node in next_nodes])
        nodes_to_remove.extend(next_nodes)

        if len(next_nodes) == 1:
          # add element-wise nodes
          n = find_node_index_by_name(graph, next_nodes[0].name)
          outputs_new = []
          for i in range(2):
            ew_node_output = f"token_{gvn()}"
            ew_node = onnx.helper.make_node(
              next_nodes[0].op_type,
              name=f"{next_nodes[0].op_type}_{gvn()}_",
              inputs=[outputs[i]] + next_nodes[0].input[1:],
              outputs=[ew_node_output],
            )
            graph.node.insert(n + i, ew_node)
            outputs_new.append(ew_node_output)
            self.prev_conv2 = ew_node
          outputs = outputs_new
        elif len(next_nodes) == 2:
          n = find_node_index_by_name(graph, next_nodes[-1].name)
          outputs_new = []
          for i in range(2):
            sigmoid_node_output = f"token_{gvn()}"
            sigmoid_node = onnx.helper.make_node(
              silu_like_type(graph, next_nodes),
              name=f"Sigmoid_{gvn()}_",
              inputs=[outputs[i]],
              outputs=[sigmoid_node_output],
            )
            graph.node.insert(n + 2*i, sigmoid_node)

            mul_node_output = f"token_{gvn()}"
            mul_node = onnx.helper.make_node(
              "Mul",
              name=f"Mul_{gvn()}_",
              inputs=[outputs[i], sigmoid_node_output],
              outputs=[mul_node_output],
            )
            graph.node.insert(n + 2*i + 1, mul_node)
            outputs_new.append(mul_node_output)
            self.prev_conv2 = mul_node
          outputs = outputs_new

        prev_node = next_nodes[-1]

        next_nodes_new = find_nodes_by_arg_name(graph, next_nodes[-1].output[0])
        next_nodes_new.remove(next_nodes[-1])
        next_nodes = next_nodes_new

      # add concat node
      concat_node_output = f"token_{gvn()}"
      concat_node = onnx.helper.make_node(
        'Concat',
        name=f'Concat_{gvn()}_',
        inputs=outputs,
        outputs=[concat_node_output],
        axis=2
      )

      n = max([max([
        find_node_index_by_name(graph, node.name)
          for node in find_nodes_by_arg_name(graph, output)])
            for output in outputs])
      graph.node.insert(n+1, concat_node)

      for n in next_nodes:
        for i in range(len(n.input)):
          if n.input[i] == prev_node.output[0]:
            n.input[i] = concat_node_output

      graph.node.remove(node)
      self.outputs = outputs

    for n in nodes_to_remove:
      graph.node.remove(n)

  def transform(self, model, nodes, stage=2, is_gpu_first=True):
    if self.satisfy(model.graph, nodes):
      for i in range(len(nodes)):
        self.apply(model.graph, nodes, stage, i, is_gpu_first)
        model = onnx.shape_inference.infer_shapes(model)
      par_exec_id(True)
      self.reset()
    return model

# TODO: fix pipeline stage granularity search
class PipelineMultipleStage:
  def __init__(self, onnx_datatype=onnx_datatype):
    self.prev_convs = []
    self.extra_slice = 0
    self.outputs = []
    self.ONNX_DATATYPE = onnx_datatype

  def reset(self):
    self.prev_convs = []
    self.extra_slice = 0
    self.outputs = []

  def satisfy(self, graph, nodes):
    return True

  def get_h_gpu(self, graph, node, div=2):
    return get_arg_shape(graph, node, node.input[0])[2] // div

  def postfix(self, nodes, i, is_gpu_first):
    conv_nodes = filter(lambda x: x.startswith("Conv"), nodes[:i+1])
    n = len(list(conv_nodes))
    if n % 2 == 1 and is_gpu_first or n % 2 == 0 and not is_gpu_first:
      return "_added"
    else:
      return "_pim_added"

  def apply(self, graph, nodes, stage, i, is_gpu_first):
    nodes_to_remove = []
    if i == 0:
      node_index = find_node_index_by_name(graph, nodes[0])
      node = graph.node[node_index]

      weight = find_initializer_by_arg_name(graph, node.input[1])
      bias = None
      if len(node.input) > 2:
        bias = find_initializer_by_arg_name(graph, node.input[2])

      # add split node
      input_arg = node.input[0]
      if graph.node[0] != node:
        input_nodes = find_nodes_by_arg_name(graph, node.input[0])
        input_nodes.remove(node)
        input_arg = input_nodes[0].output[0]

      h = get_arg_shape(graph, node, node.input[0])[2]
      kernel_shape = find_attribute_by_name(node, 'kernel_shape').ints
      h_gpu = self.get_h_gpu(graph, node, stage)
      pads = find_attribute_by_name(node, 'pads').ints
      assert pads[0] == pads[2]

      strides = find_attribute_by_name(node, 'strides').ints

      outputs = []
      for s in range(stage):
        ps_stride = (h_gpu + 1 + s) % strides[0]
        if s == 0:
          ps_stride = 0

        slice_start_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=slice_start_name,
          tensor_array=np.array([max(h_gpu * s - kernel_shape[0] // 2 + ps_stride, 0)]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)

        h_gpu_real = h_gpu * (s + 1)
        if s == stage - 1:
          h_gpu_real = h
        slice_end_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=slice_end_name,
          tensor_array=np.array([min(h_gpu_real + kernel_shape[0] // 2, h)]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)

        slice_axis_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=slice_axis_name,
          tensor_array=np.array([2]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)

        split_output = f"token_{gvn()}"
        split_node = onnx.helper.make_node(
          'Slice',
          name=f'Slice_{gvn()}_',
          inputs=[input_arg, slice_start_name, slice_end_name, slice_axis_name],
          outputs=[split_output],
        )
        n = find_node_index_by_name(graph, node.name)
        graph.node.insert(n, split_node)

        pad_start = 0
        if s == 0:
          pad_start = pads[0]
        pad_end = 0
        if s == stage - 1:
          pad_end = pads[2]
        conv_arg = f"token_{gvn()}"
        pads_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=pads_name,
          tensor_array=np.array([0, 0, pad_start, 0, 0, 0, pad_end, 0]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)
        pad_node = onnx.helper.make_node(
          'Pad',
          name=f'Pad_{gvn()}_',
          inputs=[split_output, pads_name],
          outputs=[conv_arg],
        )
        graph.node.insert(n+1, pad_node)

        # add conv node
        conv_weight_name = f"token_{gvn()}"
        conv_weight = create_initializer_tensor(
          name=conv_weight_name,
          tensor_array=numpy_helper.to_array(weight), # use the same weight
          data_type=self.ONNX_DATATYPE)
        graph.initializer.append(conv_weight)

        conv_bias_name = None
        if bias is not None:
          conv_bias_name = f"token_{gvn()}"
          conv_bias = create_initializer_tensor(
            name=conv_bias_name,
            tensor_array=numpy_helper.to_array(bias),
            data_type=self.ONNX_DATATYPE)
          graph.initializer.append(conv_bias)

        inputs = [conv_arg, conv_weight_name]
        if bias is not None:
          inputs.append(conv_bias_name)

        conv_output = f"token_{gvn()}"
        conv = onnx.helper.make_node(
          'Conv',
          name=f'Conv_{gvn()}{self.postfix(nodes, i, is_gpu_first)}',
          inputs=inputs,
          outputs=[conv_output],
          dilations=find_attribute_by_name(node, 'dilations').ints,
          group=find_attribute_by_name(node, 'group').i,
          kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
          pads=[0, pads[1], 0, pads[3]],
          strides=find_attribute_by_name(node, 'strides').ints,
        )
        graph.node.insert(n+1+s, conv)
        outputs.append(conv_output)

        next_nodes = find_nodes_by_arg_name(graph, node.output[0])
        next_nodes.remove(node)

      # last activation
      prev_node = node
      if supportedActivationNodes(graph, next_nodes):
        print([node.name for node in next_nodes])
        nodes_to_remove.extend(next_nodes)

        if len(next_nodes) == 1:
          # add element-wise nodes
          n = find_node_index_by_name(graph, next_nodes[0].name)
          outputs_new = []
          for i in range(stage):
            ew_node_output = f"token_{gvn()}"
            ew_node = onnx.helper.make_node(
              next_nodes[0].op_type,
              name=f"{next_nodes[0].op_type}_{gvn()}_",
              inputs=[outputs[i]] + next_nodes[0].input[1:],
              outputs=[ew_node_output],
            )
            graph.node.insert(n + i, ew_node)
            outputs_new.append(ew_node_output)
            self.prev_convs.append(ew_node)
          outputs = outputs_new
        elif len(next_nodes) == 2:
          n = find_node_index_by_name(graph, next_nodes[-1].name)
          outputs_new = []
          for i in range(stage):
            sigmoid_node_output = f"token_{gvn()}"
            sigmoid_node = onnx.helper.make_node(
              silu_like_type(graph, next_nodes),
              name=f"Sigmoid_{gvn()}_",
              inputs=[outputs[i]],
              outputs=[sigmoid_node_output],
            )
            graph.node.insert(n + 2*i, sigmoid_node)

            mul_node_output = f"token_{gvn()}"
            mul_node = onnx.helper.make_node(
              "Mul",
              name=f"Mul_{gvn()}_",
              inputs=[outputs[i], sigmoid_node_output],
              outputs=[mul_node_output],
            )
            graph.node.insert(n + 2*i + 1, mul_node)
            outputs_new.append(mul_node_output)
            self.prev_convs.append(mul_node)
          outputs = outputs_new

        prev_node = next_nodes[-1]

        next_nodes_new = find_nodes_by_arg_name(graph, next_nodes[-1].output[0])
        next_nodes_new.remove(next_nodes[-1])
        next_nodes = next_nodes_new

      # add concat node
      concat_node_output = f"token_{gvn()}"
      concat_node = onnx.helper.make_node(
        'Concat',
        name=f'Concat_{gvn()}_',
        inputs=outputs,
        outputs=[concat_node_output],
        axis=2
      )
      n = max([max([
        find_node_index_by_name(graph, node.name)
          for node in find_nodes_by_arg_name(graph, output)])
            for output in outputs])
      graph.node.insert(n+1, concat_node)

      for n in next_nodes:
        for i in range(len(n.input)):
          if n.input[i] == prev_node.output[0]:
            n.input[i] = concat_node_output

      graph.node.remove(node)
      self.outputs = outputs
    else:
      node_index = find_node_index_by_name(graph, nodes[i])
      node = graph.node[node_index]

      weight = find_initializer_by_arg_name(graph, node.input[1])
      bias = None
      if len(node.input) > 2:
        bias = find_initializer_by_arg_name(graph, node.input[2])

      # add split node
      input_arg = node.input[0]
      if graph.node[0] != node:
        input_nodes = find_nodes_by_arg_name(graph, node.input[0])
        input_nodes.remove(node)
        input_arg = input_nodes[0].output[0]

      h = get_arg_shape(graph, node, node.input[0])[2]
      kernel_shape = find_attribute_by_name(node, 'kernel_shape').ints
      h_gpu = self.get_h_gpu(graph, node, stage)
      pads = find_attribute_by_name(node, 'pads').ints
      assert pads[0] == pads[2]

      outputs = []
      for s in range(stage):
        slice_start_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=slice_start_name,
          tensor_array=np.array([max(h_gpu * s - kernel_shape[0] // 2, 0)]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)

        h_gpu_real = h_gpu * (s + 1)
        if s == stage - 1:
          h_gpu_real = h
        slice_end_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=slice_end_name,
          tensor_array=np.array([min(h_gpu_real + kernel_shape[0] // 2, h)]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)

        slice_axis_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=slice_axis_name,
          tensor_array=np.array([2]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)

        split_output = f"token_{gvn()}"
        split_node = onnx.helper.make_node(
          'Slice',
          name=f'Slice_{gvn()}_',
          inputs=[input_arg, slice_start_name, slice_end_name, slice_axis_name],
          outputs=[split_output],
        )
        n = find_node_index_by_name(graph, node.name)
        graph.node.insert(n, split_node)

        pad_start = 0
        if s == 0:
          pad_start = pads[0]
        pad_end = 0
        if s == stage - 1:
          pad_end = pads[2]
        conv_arg = f"token_{gvn()}"
        pads_name = f"token_{gvn()}"
        tensor_ = create_initializer_tensor(
          name=pads_name,
          tensor_array=np.array([0, 0, pad_start, 0, 0, 0, pad_end, 0]),
          data_type=onnx.TensorProto.INT64)
        graph.initializer.append(tensor_)
        pad_node = onnx.helper.make_node(
          'Pad',
          name=f'Pad_{gvn()}_',
          inputs=[split_output, pads_name],
          outputs=[conv_arg],
        )
        graph.node.insert(n+1, pad_node)

        # add conv node
        conv_weight_name = f"token_{gvn()}"
        conv_weight = create_initializer_tensor(
          name=conv_weight_name,
          tensor_array=numpy_helper.to_array(weight), # use the same weight
          data_type=self.ONNX_DATATYPE)
        graph.initializer.append(conv_weight)

        conv_bias_name = None
        if bias is not None:
          conv_bias_name = f"token_{gvn()}"
          conv_bias = create_initializer_tensor(
            name=conv_bias_name,
            tensor_array=numpy_helper.to_array(bias),
            data_type=self.ONNX_DATATYPE)
          graph.initializer.append(conv_bias)

        inputs = [conv_arg, conv_weight_name]
        if bias is not None:
          inputs.append(conv_bias_name)

        conv_output = f"token_{gvn()}"
        conv = onnx.helper.make_node(
          'Conv',
          name=f'Conv_{gvn()}{self.postfix(nodes, i, is_gpu_first)}',
          inputs=inputs,
          outputs=[conv_output],
          dilations=find_attribute_by_name(node, 'dilations').ints,
          group=find_attribute_by_name(node, 'group').i,
          kernel_shape=find_attribute_by_name(node, 'kernel_shape').ints,
          pads=[0, pads[1], 0, pads[3]],
          strides=find_attribute_by_name(node, 'strides').ints,
        )
        graph.node.insert(n+1+s, conv)
        outputs.append(conv_output)

        next_nodes = find_nodes_by_arg_name(graph, node.output[0])
        next_nodes.remove(node)

      # last activation
      prev_node = node
      if supportedActivationNodes(graph, next_nodes):
        print([node.name for node in next_nodes])
        nodes_to_remove.extend(next_nodes)

        if len(next_nodes) == 1:
          # add element-wise nodes
          n = find_node_index_by_name(graph, next_nodes[0].name)
          outputs_new = []
          for i in range(stage):
            ew_node_output = f"token_{gvn()}"
            ew_node = onnx.helper.make_node(
              next_nodes[0].op_type,
              name=f"{next_nodes[0].op_type}_{gvn()}_",
              inputs=[outputs[i]] + next_nodes[0].input[1:],
              outputs=[ew_node_output],
            )
            graph.node.insert(n + i, ew_node)
            outputs_new.append(ew_node_output)
            self.prev_convs.append(ew_node)
          outputs = outputs_new
        elif len(next_nodes) == 2:
          n = find_node_index_by_name(graph, next_nodes[-1].name)
          outputs_new = []
          for i in range(stage):
            sigmoid_node_output = f"token_{gvn()}"
            sigmoid_node = onnx.helper.make_node(
              silu_like_type(graph, next_nodes),
              name=f"Sigmoid_{gvn()}_",
              inputs=[outputs[i]],
              outputs=[sigmoid_node_output],
            )
            graph.node.insert(n + 2*i, sigmoid_node)

            mul_node_output = f"token_{gvn()}"
            mul_node = onnx.helper.make_node(
              "Mul",
              name=f"Mul_{gvn()}_",
              inputs=[outputs[i], sigmoid_node_output],
              outputs=[mul_node_output],
            )
            graph.node.insert(n + 2*i + 1, mul_node)
            outputs_new.append(mul_node_output)
            self.prev_convs.append(ew_node)
          outputs = outputs_new

        prev_node = next_nodes[-1]

        next_nodes_new = find_nodes_by_arg_name(graph, next_nodes[-1].output[0])
        next_nodes_new.remove(next_nodes[-1])
        next_nodes = next_nodes_new

      # add concat node
      concat_node_output = f"token_{gvn()}"
      concat_node = onnx.helper.make_node(
        'Concat',
        name=f'Concat_{gvn()}_',
        inputs=outputs,
        outputs=[concat_node_output],
        axis=2
      )

      n = max([max([
        find_node_index_by_name(graph, node.name)
          for node in find_nodes_by_arg_name(graph, output)])
            for output in outputs])
      graph.node.insert(n+1, concat_node)

      for n in next_nodes:
        for i in range(len(n.input)):
          if n.input[i] == prev_node.output[0]:
            n.input[i] = concat_node_output

      graph.node.remove(node)
      self.outputs = outputs

    for n in nodes_to_remove:
      graph.node.remove(n)

  def transform(self, model, nodes, stage=2, is_gpu_first=True):
    if self.satisfy(model.graph, nodes):
      for i in range(len(nodes)):
        self.apply(model.graph, nodes, stage, i, is_gpu_first)
        model = onnx.shape_inference.infer_shapes(model)
        model = onnx.shape_inference.infer_shapes(model)
      self.reset()
    return model

class OffloadFC(TransformerBase):
  def reset(self):
    pass

  def satisfy(self, graph, node):
    # already processed
    if node.name.endswith("_offloaded"):
      return False

    if node.op_type == "Conv":
      weight = find_initializer_by_arg_name(graph, node.input[1])
      H = get_arg_shape(graph, node, node.input[0])[2]
      W = get_arg_shape(graph, node, node.input[0])[3]
      if H == 1 and W == 1 and weight.dims[2] == 1 and weight.dims[3] == 1:
        return True

    if node.op_type == "Gemm":
      A = get_arg_shape(graph, node, node.input[0])
      B = find_initializer_by_arg_name(graph, node.input[1])
      alpha = find_attribute_by_name(node, 'alpha').f
      beta = find_attribute_by_name(node, 'beta').f
      transB = find_attribute_by_name(node, 'transB').i

      if B is None:
        return False

      assert np.isclose(alpha, 1.0)
      assert np.isclose(beta, 1.0)
      assert transB == 0 or transB == 1

      return A[0] == 1

    if node.op_type == "MatMul":
      A = get_arg_shape(graph, node, node.input[0])
      B = find_initializer_by_arg_name(graph, node.input[1])

      if B is None:
        return False

      return A[0] == 1

    return False

  def apply(self, graph, node):
    node.name = f"{node.name}_{par_exec_id()}_offloaded"

  def transform(self, model):
    while True:
      inferred_model = None
      for node in model.graph.node:
        print(node.name)
        if self.satisfy(model.graph, node):
          self.apply(model.graph, node)
          self.reset()
          print(MessageToDict(node))
          inferred_model = onnx.shape_inference.infer_shapes(model)
          par_exec_id(True)
          break

      if inferred_model is None:
        break

      model = inferred_model
    return model
