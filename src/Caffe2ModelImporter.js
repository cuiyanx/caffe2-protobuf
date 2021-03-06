class Caffe2ModelImporter {
  constructor (kwargs) {
    this._isQuantized = kwargs.isQuantized;
    this._netModel = kwargs.rawModel;
    this._weightModel = kwargs.weightModel;
    this._inputSize = kwargs.inputSize;
    this._model = null;
    this._compilation = null;
    this._execution = null;
    this._tensorIds = [];         //{name: ID}
    this._tensorTypes = [];       //{ID: type}
    this._operations = [];        //{[opCode, inputs, outputs]}
    this._operands = [];          //{ID: value}
    this._quantParams = [];       //{ID: type}
    this._netMap = [];
    this._requiredOps = new Set();
    this._options = {
      softmax: kwargs.softmax,
    };
    this._operandIndex = 0;
    this._backend = kwargs.backend;
    this._prefer = kwargs.prefer;
    this._inputScaleFactor = kwargs.inputScaleFactor;
    let nnNative = {
      FLOAT32: 0,
      INT32: 1,
      UINT32: 2,
      TENSOR_FLOAT32: 3,
      TENSOR_INT32: 4,
      TENSOR_QUANT8_ASYMM: 5,
      TENSOR_QUANT8_SYMM_PER_CHANNEL: 11,
      TENSOR_QUANT8_ASYMM_SIGNED: 14,
      AVERAGE_POOL_2D: 1,
      CONV_2D: 3,
      DEPTHWISE_CONV_2D: 4,
      SOFTMAX: 25
    }; // delete
    let nnPolyfill = 22; // delete
    if (this._backend === 'WebML') {
      if (nnNative === null) {
        throw Error('Fails to initialize neural network context');
      }
      this._nn = nnNative;
    } else if (this._backend === 'WASM' || this._backend === 'WebGL') {
      this._nn = nnPolyfill;
    }
    this._bEagerMode = false;
    this._supportedOps = new Set();
  }

  setEagerMode (flag) {
    this._bEagerMode = flag;
  };

  setSupportedOps (ops) {
    this._supportedOps = ops;
  };

  async createCompiledModel () {
    let options = {
      backend: this._backend,
      eager: this._bEagerMode,
      supportedOps: this._supportedOps,
    };
    // this._model = await this._nn.createModel(options);
    let Caffe2ModelUtils = require("./Caffe2ModelUtils.js");
    let utils = new Caffe2ModelUtils(this._netModel, this._weightModel, this._isQuantized);
    this._netMap = [...utils.getCaffe2Model()];
    //console.log(this._netMap[27]);
    this._setInputTensor();
    this._addOperandsAndArgs();

    /* For integrate
    await this._model.finish();
    this._compilation = await this._model.createCompilation();

    let start = performance.now();
    this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
    await this._compilation.finish();
    this._execution = await this._compilation.createExecution();
    let elapsed = performance.now() - start;
    console.log(`compilation time: ${elapsed.toFixed(2)} ms`);
    */
  }

  async compute (inputTensors, outputTensors) {
    inputTensors.forEach((inputTensor, i) => {
      this._execution.setInput(i, inputTensor);
    });
    outputTensors.forEach((outputTensor, i) => {
      this._execution.setOutput(i, outputTensor);
    });

    let error = await this._execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  async * layerIterator (inputTensors, layerList) {
    const graph = this._netModel;

    const getLayerOutput = async (lastNodeIdx) => {
      this._tensorIds = [];
      this._tensorTypes = [];
      this._operations = [];
      this._operands = [];
      this._operandIndex = 0;
      if (this._backend !== 'WebML' && this._compilation) {
        this._compilation._preparedModel._deleteAll();
      }

      this._model = await this._nn.createModel({backend: this._backend});
      this._addOperands();
      this._addOpsAndParams(); // Realization

      const lastNode = graph.op[lastNodeIdx];
      const output = lastNode.output[0];
      const input = lastNode.input;
      const inputIds = this._getTensorsId(input); // Realization
      const outputIds = this._getTensorsId(output); // Realization
      this._model.identifyInputsAndOutputs(inputIds, outputIds);

      await this._model.finish();
      this._compilation = await this._model.createCompilation();
      this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
      await this._compilation.finish();
      this._execution = await this._compilation.createExecution();

      const outputSize = output.shape().reduce((a, b) => a * b); // Realization
      let outputTensor;
      if (this._isQuantized) {
        outputTensor = new Uint8Array(outputSize);
      } else {
        outputTensor = new Float32Array(outputSize);
      }
      await this.compute(inputTensors, [outputTensor]);
      return {
        layerId: lastNodeIdx,
        outputName: lastNode.name,
        tensor: outputTensor,
        outputIds: outputIds,
        inputIds: inputIds
      };
    };

    const operatorsLength = graph.op.length;
    if (typeof layerList === 'undefined') {
      for (let lastNode = 0; lastNode < operatorsLength;) {
        const layerOutput = await getLayerOutput(lastNode);
        yield layerOutput;
        lastNode = layerOutput.layerId + 1;
      }
    } else {
      for (let layerId of layerList) {
        if (layerId >= operatorsLength || layerId < 0) {
          throw new Error(`Illegal layer ${layerId}`);
        }
        yield await getLayerOutput(layerId);
      }
    }
  }

  _setInputTensor () {
    let inputName = this._netMap[0].input[0].name;
    let inputDims = this._inputSize;
    let inputType;
    if (this._isQuantized) {
      inputType = {
        type: this._nn.TENSOR_QUANT8_ASYMM_SIGNED,
        dimensions: inputDims,
        scale: 0,
        zeroPoint: 0
      };
    }
    this._addTensor(inputName, inputType);
  }

  _addTensor (name, type, value) {
    let index = this._addOperand(type, value);
    this._tensorIds[name] = index;
    return index;
  }

  _addArgInt32 (value) {
    return this._addOperand({type: this._nn.INT32}, new Int32Array(value));
  }

  _addArgFloat32 (value) {
    return this._addOperand({type: this._nn.FLOAT32}, new Float32Array(value));
  }

  _addOperand (type, value) {
    let index = this._operandIndex++;
    // Cache operand type
    this._tensorTypes.push(type);
    if (typeof value !== 'undefined') {
      this._setOperandValue(index, value);
    }
    return index;
  }

  _addOperation(opCode, inputs, outputs) {
    console.log(`  opCode: ${opCode}`);
    console.log(`  inputs: [${inputs}], outputs: [${outputs}]`);
    // Cache operaion. It depends on operands that have not yet been added
    this._operations.push([opCode, inputs, outputs]);
    this._requiredOps.add(opCode);
  }

  _setOperandValue (index, value) {
    // Cache operand value
    this._operands[index] = value;
  }

  _getAttributeName (tensor) {
    return tensor["name"];
  }

  _getAttributeValue (tensor, keyword) {
    return tensor[keyword]["value"];
  }

  _getAttributeType (tensor, keyword) {
    return tensor[keyword]["type"];
  }

  _getTensorIdByName (name) {
    let index = this._tensorIds[name];
    if (typeof index === 'undefined')
      throw new Error(`Tensor ${name} is not found`);
    return index;
  }

  _getTensorTypeByName (name) {
    let index = this._tensorIds[name];
    if (typeof index === 'undefined')
      throw new Error(`Tensor ${name} is not found`);
    return this._tensorTypes[index];
  }

  _getFuseCode(max, min) {
    if (max === 6 && min === 0) {
      return this._nn.FUSED_RELU6;
    } else if (max === 1 && min === -1) {
      return this._nn.FUSED_RELU1;
    } else {
      return this._nn.FUSED_NONE;
    }
  }

  // Add operands
  _addOperandsAndArgs() {
    for (let nodeIdx in this._netMap) {
      let node = this._netMap[nodeIdx];
      console.log(`${node.operator} (${node.name})`);
      let opCode;
      let inputs = [];
      let outputs = [];
      switch(node.operator) {
        case "Int8Conv":
        case "Int8ConvRelu": {
          // Add inputs
          let inputTensor = node.input[0];
          let filterTensor = node.input[1];
          let biasTensor = node.input[2];
          let args = node.arg;

          // Input
          let inputName = this._getAttributeName(inputTensor);
          let inputType = this._getTensorTypeByName(inputName);
          let inputDime = inputType.dimensions;
          let inputTypeCode = inputType.type;
          let inputPoint = inputType.zeroPoint;
          let inputScales = inputType.scale;
          console.log(`  input shape: [${inputDime}]`);

          // Filter
          let filterName = this._getAttributeName(filterTensor);
          let filterDims = this._getAttributeValue(filterTensor, "shape");
          let filterValue = this._getAttributeValue(filterTensor, "values");
          let filterDataType = this._getAttributeType(filterTensor, "values");
          let filterPoint = this._getAttributeValue(filterTensor, "Y_zero_point");
          let filterScales = this._getAttributeValue(filterTensor, "Y_scales");
          let filterTypeCode = null;
          let isPerChannel = false;
          if (filterScales.length > 1) {
            filterTypeCode = this._nn.TENSOR_QUANT8_SYMM_PER_CHANNEL;
            isPerChannel = true;
          } else {
            filterTypeCode = inputTypeCode;
          }

          // Bias
          let biasName = this._getAttributeName(biasTensor);
          let biasDims = this._getAttributeValue(biasTensor, "shape");
          let biasValue = this._getAttributeValue(biasTensor, "values");
          let biasDataType = this._getAttributeType(biasTensor, "values");
          let biasPoint = this._getAttributeValue(biasTensor, "Y_zero_point");
          let biasScales = this._getAttributeValue(biasTensor, "Y_scales");
          let biasTypeCode = this._nn.TENSOR_INT32;
          let biasType = null;
          if (isPerChannel) {
            biasType = {
              type: biasTypeCode,
              dimensions: biasDims
            };
          } else {
            biasType = {
              type: biasTypeCode,
              dimensions: biasDims,
              scale: biasScales,
              zeroPoint: biasPoint
            };
          }
          console.log(`  bias shape: [${biasDims}]`);

          let kernels = this._getAttributeValue(args, "kernels");
          if (!kernels || kernels.length !== 2)
            throw new Error('Invalid kernels');
          let kernelHeight = kernels[0];
          let kernelWidth = kernels[1];

          // Pad
          let pads = this._getAttributeValue(args, "pads");
          let [paddingLeft, paddingRight, paddingTop, paddingBottom] = pads;
          console.log(`  pads: [${pads}]`);

          // Stride
          let strides = this._getAttributeValue(args, "strides");
          let [strideWidth, strideHeight] = strides;
          console.log(`  strides: [${strides}]`);

          // Group
          let isDepthWiseConv = false;
          let group = 0;
          let inputChannel = inputDime[inputDime.length - 1];
          if (args.hasOwnProperty("group")) {
            group = this._getAttributeValue(args, "group")[0];
          }

          // Fuse Relu
          let boundMax = 0;
          let boundMin = 0;
          if (args.hasOwnProperty("bound_max") && args.hasOwnProperty("bound_min")) {
            boundMax = this._getAttributeValue(args, "bound_max");
            boundMin = this._getAttributeValue(args, "bound_min");
            console.log(`  bound: [${boundMax}, ${boundMin}]`);
          }
          let fuseCode = this._getFuseCode(boundMax, boundMin);

          if (group > 1) {
            if (group !== inputChannel) {
              throw new Error('Group convolution is not supported.');
            } else {
              isDepthWiseConv = true;
              console.log(`  group: ${group} (depthwise convolution)`);
              let nhwcData = filterValue;
              let chwnData = new Int8Array(nhwcData.length);
              let N = filterDims[0];
              let H = filterDims[1];
              let W = filterDims[2];
              // NHWC -> CHWN where C === 1
              for (let n = 0; n < N; ++n) {
                for (let h = 0; h < H; ++h) {
                  for (let w = 0; w < W; ++w) {
                    chwnData[h*W*N + w*N + n] = nhwcData[n*H*W + h*W + w];
                  }
                }
              }
              filterValue = chwnData;
              filterDims[0] = 1;
              filterDims[3] = group;
            }
          }
          console.log(`  filter shape: [${filterDims}]`);

          let filterType = {};
          if (isPerChannel) {
            filterType = {
              type: filterTypeCode,
              dimensions: filterDims
            };
          } else {
            filterType = {
              type: filterTypeCode,
              dimensions: filterDims,
              scale: filterScales,
              zeroPoint: filterPoint
            };
          }

          inputs.push(this._getTensorIdByName(inputName));
          inputs.push(this._addTensor(filterName, filterType, filterValue));
          let filterID = this._getTensorIdByName(filterName);
          let channelDim = 1;
          if (isPerChannel) {
            if (isDepthWiseConv) {
              channelDim = 3;
            }
            this._quantParams[filterID] = {
              channelDim: channelDim,
              scales: new Float32Array([filterScales])
            };
          }
          inputs.push(this._addTensor(biasName, biasType, biasValue));
          inputs.push(this._addArgInt32([paddingLeft]));
          inputs.push(this._addArgInt32([paddingRight]));
          inputs.push(this._addArgInt32([paddingTop]));
          inputs.push(this._addArgInt32([paddingBottom]));
          inputs.push(this._addArgInt32([strideWidth]));
          inputs.push(this._addArgInt32([strideHeight]));
          if (isDepthWiseConv) {
            inputs.push(this._addArgInt32([1]));
          }
          inputs.push(this._addArgInt32([fuseCode]));

          // Add outputs
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          let outputTypeCode = inputTypeCode;
          let outputDims = [
            inputDime[0],
            Math.floor((inputDime[1] - kernelHeight + paddingLeft + paddingTop) / strideHeight + 1),
            Math.floor((inputDime[2] - kernelWidth + paddingRight + paddingBottom) / strideWidth + 1),
            biasDims[0]
          ];
          let outputScales = 1;
          if (args.hasOwnProperty("Y_scale")) {
            outputScales = this._getAttributeValue(args, "Y_scale");
          }
          let outputPoint = 0;
          if (args.hasOwnProperty("Y_zero_point")) {
            outputPoint = this._getAttributeValue(args, "Y_zero_point");
          }
          let outputType = {
            type: outputTypeCode,
            dimensions: outputDims,
            scale: outputScales,
            zeroPoint: outputPoint
          };
          let outputID = this._addTensor(outputName, outputType);
          outputs.push(outputID);
          console.log(`  output shape: [${outputDims}]`);

          // Add operation
          opCode = isDepthWiseConv ? this._nn.DEPTHWISE_CONV_2D : this._nn.CONV_2D;
        } break;
        case "Int8AveragePool": {
          // Add inputs
          let inputTensor = node.input[0];
          let args = node.arg;

          // Input
          let inputName = this._getAttributeName(inputTensor);
          let inputType = this._getTensorTypeByName(inputName);
          let inputDime = inputType.dimensions;
          let inputTypeCode = inputType.type;
          let inputPoint = inputType.zeroPoint;
          let inputScales = inputType.scale;
          console.log(`  input shape: [${inputDime}]`);

          // Pad
          let pads = [0, 0, 0, 0];
          if (args.hasOwnProperty("pads")) {
            pads = this._getAttributeValue(args, "pads");
          }
          let [paddingLeft, paddingRight, paddingTop, paddingBottom] = pads;
          console.log(`  pads: [${pads}]`);

          // Stride
          let strides = this._getAttributeValue(args, "strides");
          let [strideWidth, strideHeight] = strides;
          console.log(`  strides: [${strides}]`);

          // Filter
          let filter = this._getAttributeValue(args, "kernels");
          if (!filter || filter.length !== 2)
            throw new Error('Invalid filter');
          let [filterWidth, filterHeight] = filter;
          console.log(`  filter: [${filter}]`);

          // Fuse Relu
          let boundMax = 0;
          let boundMin = 0;
          if (args.hasOwnProperty("bound_max") && args.hasOwnProperty("bound_min")) {
            boundMax = this._getAttributeValue(args, "bound_max");
            boundMin = this._getAttributeValue(args, "bound_min");
            console.log(`  bound: [${boundMax}, ${boundMin}]`);
          }
          let fuseCode = this._getFuseCode(boundMax, boundMin);

          inputs.push(this._getTensorIdByName(inputName));
          inputs.push(this._addArgInt32([paddingLeft]));
          inputs.push(this._addArgInt32([paddingRight]));
          inputs.push(this._addArgInt32([paddingTop]));
          inputs.push(this._addArgInt32([paddingBottom]));
          inputs.push(this._addArgInt32([strideWidth]));
          inputs.push(this._addArgInt32([strideHeight]));
          inputs.push(this._addArgInt32([filterWidth]));
          inputs.push(this._addArgInt32([filterHeight]));
          inputs.push(this._addArgInt32([fuseCode]));

          // Add outputs
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          let outputTypeCode = inputTypeCode;
          let outputDims = [
            inputDime[0],
            Math.floor((inputDime[1] - filterHeight + paddingLeft + paddingTop) / strideHeight + 1),
            Math.floor((inputDime[2] - filterWidth + paddingRight + paddingBottom) / strideWidth + 1),
            inputDime[3]
          ];
          let outputScales = inputScales;
          if (args.hasOwnProperty("Y_scale")) {
            outputScales = this._getAttributeValue(args, "Y_scale");
          }
          let outputPoint = inputPoint;
          if (args.hasOwnProperty("Y_zero_point")) {
            outputPoint = this._getAttributeValue(args, "Y_zero_point");
          }
          let outputType = {
            type: outputTypeCode,
            dimensions: outputDims,
            scale: outputScales,
            zeroPoint: outputPoint
          };
          let outputID = this._addTensor(outputName, outputType);
          outputs.push(outputID);
          console.log(`  output shape: [${outputDims}]`);

          // Add operation
          opCode = this._nn.AVERAGE_POOL_2D;
        } break;
        case "Softmax": {
          // Add inputs
          let inputTensor = node.input[0];
          let args = node.arg;

          // Input
          let inputName = this._getAttributeName(inputTensor);
          let inputType = this._getTensorTypeByName(inputName);
          let inputDime = inputType.dimensions;
          let inputTypeCode = inputType.type;
          let inputPoint = inputType.zeroPoint;
          let inputScales = inputType.scale;
          console.log(`  input shape: [${inputDime}]`);

          // Beta
          let beta = this._getAttributeValue(args, "axis");
          console.log(`  Beta: [${beta}]`);

          inputs.push(this._getTensorIdByName(inputName));
          inputs.push(this._addArgFloat32([beta]));

          // Add outputs
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          let outputTypeCode = inputTypeCode;
          let outputDims = inputDime;
          let outputScales = 1;
          if (args.hasOwnProperty("Y_scale")) {
            outputScales = this._getAttributeValue(args, "Y_scale");
          }
          let outputPoint = 0;
          if (args.hasOwnProperty("Y_zero_point")) {
            outputPoint = this._getAttributeValue(args, "Y_zero_point");
          }
          let outputType = {
            type: outputTypeCode,
            dimensions: outputDims,
            scale: outputScales,
            zeroPoint: outputPoint
          };
          let outputID = this._addTensor(outputName, outputType);
          outputs.push(outputID);
          console.log(`  output shape: [${outputDims}]`);

          // Add operation
          opCode = this._nn.SOFTMAX;
        } break;
        default: {
          throw new Error(`${node.operator} is not supported.`);
        }
      }

      this._addOperation(opCode, inputs, outputs);
    }
    /* For integrate
    // Write back all cached operands and operations
    for (let type of this._tensorTypes) {
      this._model.addOperand(type);
    }

    for (let [index, value] of Object.entries(this._operands)) {
      this._model.setOperandValue(index, value);
    }

    for (let [index, type] of Object.entries(this._quantParams)) {
      this._model.setOperandSymmPerChannelQuantParams(index, type);
    }

    for (let [opCode, inputs, outputs] of this._operations) {
      this._model.addOperation(opCode, inputs, outputs);
    }
    */
  }

  getRequiredOps() {
    return this._requiredOps;
  }
}

module.exports = Caffe2ModelImporter;
