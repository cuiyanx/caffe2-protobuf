class Caffe2ModelImporter {
  constructor(kwargs) {
    this._isQuantized = kwargs.isQuantized;
    this._rawModel = kwargs.rawModel;
    this._model = null;
    this._compilation = null;
    this._execution = null;
    this._tensorIds = [];
    this._tensorTypes = [];
    this._operations = [];
    this._operands = [];
    this._requiredOps = new Set();
    this._options = {
      softmax: kwargs.softmax,
    };
    this._operandIndex = 0;
    this._backend = kwargs.backend;
    this._prefer = kwargs.prefer;
    this._inputScaleFactor = kwargs.inputScaleFactor;
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

  setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  async createCompiledModel() {
    let options = {
      backend: this._backend,
      eager: this._bEagerMode,
      supportedOps: this._supportedOps,
    };
    this._model = await this._nn.createModel(options);

    this._addTensorOperands();
    this._addOpsAndParams();
    this._addInputsOutputs();

    await this._model.finish();
    this._compilation = await this._model.createCompilation();

    let start = performance.now();
    this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
    await this._compilation.finish();
    this._execution = await this._compilation.createExecution();
    let elapsed = performance.now() - start;
    console.log(`compilation time: ${elapsed.toFixed(2)} ms`);
  }

  async compute(inputTensors, outputTensors) {
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

  async * layerIterator(inputTensors, layerList) {
    let graph = this._rawModel;

    let createLayer = async (nodeIdx) => {
      this._tensorIds = [];
      this._tensorTypes = [];
      this._operations = [];
      this._operands = [];
      this._operandIndex = 0;
      if (this._backend !== 'WebML' && this._compilation) {
        this._compilation._preparedModel._deleteAll();
      }

      this._model = await this._nn.createModel({backend: this._backend});
      this._addTensorOperands(nodeIdx);
      this._addOpsAndParams(nodeIdx);

      let node = graph.op[nodeIdx];
      let output = node.output[0];
      let input = node.input[0];
      let inputIds = [this._getTensorId(input)];
      let outputIds = [this._getTensorId(output)];
      this._model.identifyInputsAndOutputs(inputIds, outputIds);

      await this._model.finish();
      this._compilation = await this._model.createCompilation();
      this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
      await this._compilation.finish();
      this._execution = await this._compilation.createExecution();

      let outputSize = output.shape().reduce((a, b) => a * b);
      let outputTensor;
      if (this._isQuantized) {
        outputTensor = new Uint8Array(outputSize);
      } else {
        outputTensor = new Float32Array(outputSize);
      }
      await this.compute(inputTensors, [outputTensor]);
      return {
        layerId: nodeIdx,
        outputName: node.name,
        tensor: outputTensor,
        outputIds: outputIds,
        inputIds: inputIds
      };
    };

    let operatorsLength = graph.op.length;
    if (typeof layerList === 'undefined') {
      for (let layerId = 0; layerId < operatorsLength;) {
        let layerInfo = await createLayer(layerId);
        yield layerInfo;
        layerId = layerInfo.layerId + 1;
      }
    } else {
      for (let layerId of layerList) {
        if (layerId >= operatorsLength || layerId < 0) {
          throw new Error(`Illegal layer ${layerId}`);
        }
        yield await createLayer(layerId);
      }
    }
  }

  _addTensorOperands(nodeIdx) {
    const graph = this._rawModel.op[nodeIdx];

    for (const input of graph.input) {
      const inputName = input.graphId();
      const scale = this._inputScaleFactor == undefined ? 1.0 : this._inputScaleFactor;
      const inputType = {
        type: this._getTypeCode(input.dataType()), dimensions: input.shape(), scale
      };
      this._addNamedOperand(inputName, inputType);
    }
  }
}
